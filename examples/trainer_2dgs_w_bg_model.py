import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    CameraOptModuleMLP,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
)
from gsplat import export_splats
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from gsplat.rendering import rasterization_2dgs
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.cuda._wrapper import spherical_harmonics
from gsplat.utils import compute_pixel_rays
from nerfview import CameraState, RenderTabState, apply_float_colormap
import evo.core.geometry as geometry
from splatfactow_field import BGField, SplatfactoWField

LOG_TINY_SCALE = math.log(1e-16)

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Format to export ply files
    export_fmt: Literal["ply", "splat", "ply_compressed"] = "ply"
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Type of camera optimization
    pose_opt_type: Literal["default", "mlp"] = "default"
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable normal consistency loss. (Currently for 2DGS only)
    normal_loss: bool = False
    # Weight for normal loss
    normal_lambda: float = 5e-2
    # Iteration to start normal consistency regulerization
    normal_start_iter: int = 7_000

    # Distortion loss. (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000
    # background model derived from splatfacto-w ===============================
    enable_bg_model: bool = False
    # Number of layers in the background model
    bg_num_layers: int = 3
    # Width of each layer in the background model
    bg_layer_width: int = 128
    # The degree of SH to use for the background model
    bg_sh_degree: int = 4
    # Dimension of the appearance embedding, if 0, no appearance embedding is used
    appearance_embed_dim: int = 48
    # Number of layers in the appearance model
    appearance_num_layers: int = 3
    # Width of each layer in the appearance model
    appearance_layer_width: int = 256
    # Whether to enable the alpha loss for punishing gaussians from occupying background space, this also works with pure color background (i.e. white for overexposed skys)
    enable_alpha_loss: bool = False
    # Dimension of the appearance feature
    appearance_features_dim: int = 72
    # ==============================================================================

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    appearance_features_dim: int = 72,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    scales[:, 2] = LOG_TINY_SCALE

    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    appearance_features = torch.zeros((N, appearance_features_dim)).float().cuda()

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("appearance_features", torch.nn.Parameter(appearance_features), 2e-2),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space and (cfg.init_type == "sfm"),
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.trainvalset = Dataset(self.parser, split="train")
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            appearance_features_dim=cfg.appearance_features_dim,
            device=self.device,
        )

        self.appearance_embeds = torch.nn.Embedding(
            len(self.trainset), cfg.appearance_embed_dim
        ).to(self.device)

        self.appearance_embeds_optimizers = [
            torch.optim.Adam(
                self.appearance_embeds.parameters(),
                lr=1e-3 * math.sqrt(cfg.batch_size),
                eps=1e-15 / math.sqrt(cfg.batch_size),
                betas=(1 - cfg.batch_size * (1 - 0.9), 1 - cfg.batch_size * (1 - 0.999)),
            )
        ]

        self.bg_model_optimizers = {}
        if cfg.enable_bg_model:
            self.bg_model = BGField(
                appearance_embedding_dim=cfg.appearance_embed_dim,
                sh_levels=cfg.bg_sh_degree,
                num_layers=cfg.bg_num_layers,
                layer_width=cfg.bg_layer_width,
                device=self.device,
            )
            bg_model_params = [
                ("bg_model_encoder", self.bg_model.encoder.parameters(), 2e-3),
                ("bg_model_sh_base", self.bg_model.sh_base_head.parameters(), 2e-3),
                ("bg_model_sh_rest", self.bg_model.sh_rest_head.parameters(), 2e-3 / 20),
            ]
            self.bg_model_optimizers = {
                name: torch.optim.Adam(
                    params,
                    lr=lr * math.sqrt(cfg.batch_size),
                    eps=1e-15 / math.sqrt(cfg.batch_size),
                    betas=(1 - cfg.batch_size * (1 - 0.9), 1 - cfg.batch_size * (1 - 0.999)),
                )
                for name, params, lr in bg_model_params
            }

        else:
            self.bg_model = None

        self.color_model_optimizers = {}
        self.color_model = SplatfactoWField(
            appearance_embed_dim=cfg.appearance_embed_dim,
            appearance_features_dim=cfg.appearance_features_dim,
            sh_levels=cfg.sh_degree,
            num_layers=cfg.appearance_num_layers,
            layer_width=cfg.appearance_layer_width,
            device=self.device,
        )
        color_model_params = [
                ("color_model_encoder", self.color_model.encoder.parameters(), 2e-3),
                ("color_model_sh_base", self.color_model.sh_base_head.parameters(), 2e-3),
                ("color_model_sh_rest", self.color_model.sh_rest_head.parameters(), 2e-3 / 20),
            ]
        self.color_model_optimizers = {
                name: torch.optim.Adam(
                    params,
                    lr=lr * math.sqrt(cfg.batch_size),
                    eps=1e-15 / math.sqrt(cfg.batch_size),
                    betas=(1 - cfg.batch_size * (1 - 0.9), 1 - cfg.batch_size * (1 - 0.999)),
                )
                for name, params, lr in color_model_params
        }

        self.cached_colors = None
        self.cached_bg_sh = None

        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.strategy = self.cfg.strategy
        self.strategy.check_sanity(self.splats, self.optimizers)

        key_for_gradient = "gradient_2dgs"

        if isinstance(self.strategy, DefaultStrategy):
            for attr in ['prune_opa', 'grow_grad2d', 'grow_scale3d', 'prune_scale3d', 'absgrad', 'revised_opacity']:
                setattr(self.strategy, attr, getattr(self.cfg, attr))
            self.strategy.key_for_gradient = key_for_gradient
            self.strategy_state = self.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.model_type = "2dgs"
            self.strategy_state = self.strategy.initialize_state()
        else:
            assert_never(self.strategy)

        self.pose_optimizers = []
        if cfg.pose_opt:
            if cfg.pose_opt_type == "default":
                self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            elif cfg.pose_opt_type == "mlp":
                self.pose_adjust = CameraOptModuleMLP(len(self.trainset)).to(self.device)
                cfg.pose_opt_lr = 15e-5
                cfg.pose_opt_reg = 0.0
            else:
                assert_never(self.cfg.pose_opt_type)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Tensor | None = None,
        training: bool = True,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        appearance_features = self.splats["appearance_features"]  # [N, F]

        image_id = kwargs.pop("image_ids", None)

        if image_id is not None:
            appearance_embed = self.appearance_embeds(
                image_id
            )
        else:
            appearance_embed = self.appearance_embeds.weight.mean(dim=0)

        if not training and self.cached_colors is not None:
            colors = self.cached_colors.detach()
        else:
            colors = self.color_model(
                appearance_embed=appearance_embed.repeat(appearance_features.shape[0], 1),
                appearance_features=appearance_features,
            ).float()
            self.cached_colors = colors

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.strategy.absgrad
                if isinstance(self.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            **kwargs,
        )

        if masks is not None:
            render_colors[~masks] = 0

        return (
            render_colors, # [..., C, height, width, X].
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        )

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["appearance_features"], gamma=5e-2 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.appearance_embeds_optimizers[0], gamma=0.3 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.color_model_optimizers["color_model_encoder"], gamma=5e-2 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.color_model_optimizers["color_model_sh_base"], gamma=5e-2 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.color_model_optimizers["color_model_sh_rest"], gamma=5e-2 / 20 ** (1.0 / max_steps)
            )
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.enable_bg_model:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.bg_model_optimizers["bg_model_encoder"], gamma=5e-2 ** (1.0 / max_steps)
                )
            )
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.bg_model_optimizers["bg_model_sh_base"], gamma=0.1 ** (1.0 / max_steps)
                )
            )
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.bg_model_optimizers["bg_model_sh_rest"], gamma=0.1 / 20 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_id = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_id)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_id)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            if cfg.enable_bg_model:
                bg_sh_degree_to_use = min(
                    step // cfg.sh_degree_interval, cfg.bg_sh_degree
                )
            else:
                bg_sh_degree_to_use = None

            # forward
            (
                renders,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_id,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB+D",
                distloss=self.cfg.dist_loss,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            appearance_embed = self.appearance_embeds(image_id)
            if cfg.enable_bg_model:
                background = self.compute_background(camtoworlds, Ks, width, height, bg_sh_degree_to_use, appearance_embed=appearance_embed)
                colors = colors + (1.0 - alphas) * background

            colors = torch.clamp(colors, 0.0, 1.0)

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            masks = data["mask"].to(device) if "mask" in data else None
            if masks is not None:
                pixels = pixels * masks[..., None]
                colors = colors * masks[..., None]

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

            # regularizations for mcmc starategy
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            if cfg.enable_bg_model and cfg.enable_alpha_loss:
                alpha_loss = torch.tensor(0.0).to(self.device)
                # bgモデルでよくあらわされている部分についてはガウシアンのalphaを小さくするように促す
                # for those pixel are well represented by bg and has low alpha, we encourage the gaussian to be transparent
                bg_mask = torch.abs(pixels - background).mean(dim=-1, keepdim=True) < 0.003
                # use a box filter to avoid penalty high frequency parts
                f = 3
                window = (torch.ones((f, f)).view(1, 1, f, f) / (f * f)).cuda()
                # マスクを平滑化
                bg_mask = (
                    torch.nn.functional.conv2d(
                        bg_mask.float().permute(0, 3, 1, 2),
                        window,
                        stride=1,
                        padding="same",
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                )
                # 平滑化後のマスク値が0.6を超えるピクセルを最終的な背景領域として判定します。
                alpha_mask = bg_mask > 0.6
                # prevent NaN
                if alpha_mask.sum() != 0: # meanの計算時に割るときに0割りを防ぐ
                    # マスクした領域のalphaが大きくならないようにペナルティをかけるloss
                    alpha_loss = alphas.squeeze(0)[alpha_mask].mean() * 0.15
                loss += alpha_loss

            loss.backward()

            if self.splats["scales"].grad is not None:
                self.splats["scales"].grad[:, 2] = 0.0


            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.normal_loss:
                    self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if isinstance(self.strategy, DefaultStrategy):
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.strategy, MCMCStrategy):
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.strategy)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.appearance_embeds_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bg_model_optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.color_model_optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    data["pose_adjust"] = self.pose_adjust.state_dict()
                torch.save(
                    data,
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                appearance_features = self.splats["appearance_features"]  # [N, F]
                appearance_embed = self.appearance_embeds(image_id)
                sh_coeffs = self.color_model(appearance_embed.repeat(appearance_features.shape[0], 1), appearance_features)
                sh0 = sh_coeffs[:, 0, :].unsqueeze(1)  # [N, 1, 3] - DC成分
                shN = sh_coeffs[:, 1:, :]  # [N, sh_dim-1, 3] - 高次成分
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format=cfg.export_fmt,
                    save_to=f"{self.ply_dir}/point_cloud_{step}.{cfg.export_fmt}",
                )


            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    # 3RGSのmlpモデルを使ってカメラ最適化をする場合は、評価用のカメラポーズも最適化する
    def optim_eval_camtoworlds(self, camtoworlds, Ks, width, height, sh_degree, near_plane, far_plane, masks, pixels, show_progress=False):
        with torch.enable_grad():
            iters = 100 #* 2
            pose_opt_lr = 8e-4
            pose_adjust = CameraOptModule(1).to(self.device)
            pose_adjust.zero_init()
            pose_optimizer = torch.optim.Adam(
                    pose_adjust.parameters(),
                    lr=pose_opt_lr,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        pose_optimizer, gamma=0.01 ** (1.0 / iters)
            )

            # Use tqdm only if show_progress is True
            iterator = tqdm.tqdm(range(iters), desc="Optimizing eval camera pose") if show_progress else range(iters)
            for _ in iterator:
                camtoworlds_optim = pose_adjust(camtoworlds, torch.tensor([0],device=self.device))
                colors, alphas, *_ = self.rasterize_splats(
                    camtoworlds=camtoworlds_optim,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    masks=masks,
                    training=False,
                )
                colors = colors[..., 0:3]
                if self.cfg.enable_bg_model:
                    background = self.compute_background(camtoworlds, Ks, width, height, self.cfg.bg_sh_degree, appearance_embed=None, training=False)
                    colors = colors + (1.0 - alphas) * background

                colors = torch.clamp(colors, 0.0, 1.0)

                #loss = F.l1_loss(renders, pixels)
                # gradient loss
                loss = compute_gradient_loss(pixels, colors)

                loss.backward()
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                # Update progress bar description only if show_progress is True
                if show_progress:
                    iterator.set_description(f"Optimizing camera pose (loss={loss.item():.6f})")

            # set gradients to none
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.zero_grad(set_to_none=True)
        return camtoworlds_optim.detach()


    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        # trainset全体の評価
        trainloader = torch.utils.data.DataLoader(
            self.trainvalset, batch_size=1, shuffle=False, num_workers=1
        )
        traineval_ellipse_time = 0
        train_metrics = {"psnr": [], "ssim": [], "lpips": []}

        camtoworlds_est, camtoworlds_gt = [], []
        for i, data in enumerate(trainloader):
            camtoworlds = data["camtoworld"].to(device)
            if "camtoworld_gt" in data:
                pesudo_gt = False
                camtoworld_gt = data["camtoworld_gt"].to(device)
            else:
                pesudo_gt = True
                camtoworld_gt = camtoworlds

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, data['image_id'].to(device))
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, data['image_id'].to(device))

            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            image_id = data["image_id"].to(device)

            camtoworlds_est.append(camtoworlds)
            camtoworlds_gt.append(camtoworld_gt)

            torch.cuda.synchronize()
            tic = time.time()
            renders, alphas, *_ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                masks=masks,
                training=False,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            traineval_ellipse_time += time.time() - tic
            colors = renders[..., 0:3]  # [1, H, W, 3]
            appearance_embed = self.appearance_embeds(image_id)
            if cfg.enable_bg_model:
                background = self.compute_background(camtoworlds, Ks, width, height, cfg.bg_sh_degree, appearance_embed=appearance_embed)
                colors = colors + (1.0 - alphas) * background

            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]


            depths = (depths - depths.min()) / (depths.max() - depths.min())

            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            train_metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            train_metrics["ssim"].append(self.ssim(colors_p, pixels_p))
            train_metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        traineval_ellipse_time /= len(trainloader)

        if cfg.pose_opt and cfg.pose_opt_type == "mlp":
            a = torch.stack(camtoworlds_est,dim=0).squeeze(1).detach().cpu().numpy()
            b = torch.stack(camtoworlds_gt,dim=0).squeeze(1).detach().cpu().numpy()
            transform = align_pose(b, a).to(device)

        # valset全体の評価
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        eval_ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            image_id = data["image_id"].to(device)

            torch.cuda.synchronize()
            tic = time.time()
            if cfg.pose_opt and cfg.pose_opt_type == "mlp":
                # まず大まかにアラインメント
                camtoworlds = torch.einsum('ij,bjk->bik', transform, camtoworlds)
                # 微調整
                camtoworlds = self.optim_eval_camtoworlds(camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=masks,
                    pixels=pixels,
                )

            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                masks=masks,
                training=False,
            )  # [1, H, W, 3]
            colors = colors[..., :3]  # Take RGB channels
            if cfg.enable_bg_model:
                background = self.compute_background(camtoworlds, Ks, width, height, cfg.bg_sh_degree, appearance_embed=None, training=False)
                colors = colors + (1.0 - alphas) * background

            colors = torch.clamp(colors, 0.0, 1.0)

            torch.cuda.synchronize()
            eval_ellipse_time += max(time.time() - tic, 1e-10)

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            # write median depths
            render_median = (render_median - render_median.min()) / (
                render_median.max() - render_median.min()
            )
            # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
            render_median = (
                apply_float_colormap(render_median).detach().cpu().squeeze(0).numpy()
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_median_depth_{step}.png",
                (render_median * 255).astype(np.uint8),
            )

            # write normals
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            normals_output = (normals * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normal_{step}.png", normals_output
            )

            # write normals from depth
            normals_from_depth *= alphas.squeeze(0).detach()
            normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
            normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                np.max(normals_from_depth) - np.min(normals_from_depth)
            )
            normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
            if len(normals_from_depth_output.shape) == 4:
                normals_from_depth_output = normals_from_depth_output.squeeze(0)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                normals_from_depth_output,
            )

            # write distortions

            render_dist = render_distort
            dist_max = torch.max(render_dist)
            dist_min = torch.min(render_dist)
            render_dist = (render_dist - dist_min) / (dist_max - dist_min)
            render_dist = (
                apply_float_colormap(render_dist).detach().cpu().squeeze(0).numpy()
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_distortions_{step}.png",
                (render_dist * 255).astype(np.uint8),
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        eval_ellipse_time /= len(valloader)

        train_psnr = torch.stack(train_metrics["psnr"]).mean()
        train_ssim = torch.stack(train_metrics["ssim"]).mean()
        train_lpips = torch.stack(train_metrics["lpips"]).mean()

        eval_psnr = torch.stack(metrics["psnr"]).mean()
        eval_ssim = torch.stack(metrics["ssim"]).mean()
        eval_lpips = torch.stack(metrics["lpips"]).mean()

        print(
            f"TRAIN PSNR: {train_psnr.item():.3f}, TRAIN SSIM: {train_ssim.item():.4f}, TRAIN LPIPS: {train_lpips.item():.3f} | "
            f"EVAL PSNR: {eval_psnr.item():.3f}, EVAL SSIM: {eval_ssim.item():.4f}, EVAL LPIPS: {eval_lpips.item():.3f} "
            f"TRAIN Time: {traineval_ellipse_time:.3f}s/image "
            f"EVAL Time: {eval_ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means'])}"
        )
        # save train stats as json
        train_stats = {
            "train_psnr": train_psnr.item(),
            "train_ssim": train_ssim.item(),
            "train_lpips": train_lpips.item(),
            "train ellipse_time": traineval_ellipse_time,
            "num_GS": len(self.splats["means"]),
        }
        with open(f"{self.stats_dir}/train_metrics_step{step:04d}.json", "w") as f:
            json.dump(train_stats, f)

        # save eval stats as json
        eval_stats = {
            "eval_psnr": eval_psnr.item(),
            "eval_ssim": eval_ssim.item(),
            "eval_lpips": eval_lpips.item(),
            "eval ellipse_time": eval_ellipse_time,
            "num_GS": len(self.splats["means"]),
        }
        with open(f"{self.stats_dir}/val_metrics_step{step:04d}.json", "w") as f:
            json.dump(eval_stats, f)

        # save stats to tensorboard
        for k, v in eval_stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, alphas, _, surf_normals, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                training=False,
            )  # [1, H, W, 4]
            colors = renders[..., 0:3]  # [1, H, W, 3]
            if cfg.enable_bg_model:
                background = self.compute_background(camtoworlds[i:i+1], K, width, height, self.cfg.bg_sh_degree, appearance_embed=None, training=False)
                colors = colors + (1.0 - alphas) * background
            colors = torch.clamp(colors, 0.0, 1.0).squeeze(0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info
        ) = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode="RGB+ED",
            training=False,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "depth":
            # normalize depth to [0, 1]
            depth = render_median
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "normal":
            render_normals = render_normals * 0.5 + 0.5  # normalize to [0, 1]
            renders = render_normals.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            render_colors = render_colors[0, ..., 0:3]
            if self.enable_bg_model:
                background = self.compute_background(c2w[None], K[None], width, height, self.cfg.bg_sh_degree, appearance_embed=None, training=False)
                render_colors = render_colors + (1.0 - render_alphas) * background
            render_colors = render_colors.clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders

    def compute_background(self, camtoworlds, Ks, width, height, bg_sh_degree_to_use, appearance_embed=None, training=True):

        directions = compute_pixel_rays(camtoworlds, Ks, width, height)
        directions = directions.view(-1, 3)  # [H*W, 3]
        if not training and self.cached_bg_sh is not None and appearance_embed is None:
            bg_sh_coeffs = self.cached_bg_sh.detach()
        else:
            bg_sh_coeffs = self.bg_model(appearance_embed)
            self.cached_bg_sh = bg_sh_coeffs
        background = spherical_harmonics(
            degrees_to_use=bg_sh_degree_to_use,
            dirs=directions,
            coeffs=bg_sh_coeffs.repeat(directions.shape[0], 1, 1),
        )
        background = background.view(1, height, width, 3)

        return background


def compute_gradient_loss(pixels, colors, edge_threshold=4, rgb_boundary_threshold=0.01):
    """
    Compute gradient-aware loss with masking

    Args:
        pixels: Target image tensor [B, H, W, C]
        colors: Rendered image tensor [B, H, W, C]
        edge_threshold: Threshold for edge detection relative to median gradient
        rgb_boundary_threshold: Threshold for RGB boundary detection
    """
    def image_gradient(image):
        # Compute image gradient using Scharr Filter
        c = image.shape[0]
        conv_y = torch.tensor(
            [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
        )
        conv_x = torch.tensor(
            [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
        )
        normalizer = 1.0 / torch.abs(conv_y).sum()
        p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
        img_grad_v = normalizer * torch.nn.functional.conv2d(
            p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
        )
        img_grad_h = normalizer * torch.nn.functional.conv2d(
            p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
        )
        return img_grad_v[0], img_grad_h[0]


    def image_gradient_mask(image, eps=0.01):
        # Compute image gradient mask
        c = image.shape[0]
        conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
        p_img = torch.abs(p_img) > eps
        img_grad_v = torch.nn.functional.conv2d(
            p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
        )
        img_grad_h = torch.nn.functional.conv2d(
            p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
        )

        return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


    # Process each batch item
    batch_losses = []
    for b in range(pixels.shape[0]):
        # Convert target image to grayscale [1, H, W]
        gray_img = pixels[b].permute(2, 0, 1).mean(dim=0, keepdim=True)

        # Compute gradients and masks
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)

        # Apply masks to gradients
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h

        # Compute gradient intensity
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        # Create edge mask based on median threshold
        median_img_grad_intensity = torch.median(img_grad_intensity)
        image_mask = (img_grad_intensity > median_img_grad_intensity * edge_threshold).float()

        # Create RGB boundary mask
        rgb_pixel_mask = (pixels[b].sum(dim=-1) > rgb_boundary_threshold).float()

        # Combine masks
        combined_mask = image_mask * rgb_pixel_mask

        # Compute masked L1 loss
        batch_loss = combined_mask * torch.abs(colors[b] - pixels[b]).mean(dim=-1)
        batch_losses.append(batch_loss.sum() / (combined_mask.sum() + 1e-8))

    # Average losses across batch
    return torch.stack(batch_losses).mean()


def align_pose(pose_a, pose_b):
    # Calculate alignment parameters using umeyama
    r, t, c = geometry.umeyama_alignment(pose_a[:,:3,3].T, pose_b[:,:3,3].T, with_scale=True)

    # Create 4x4 transformation matrix
    device = pose_a.device if torch.is_tensor(pose_a) else torch.device('cpu')
    transform = torch.eye(4, device=device)
    transform[:3,:3] = c * torch.from_numpy(r).to(device).float()  # Apply rotation and scale
    transform[:3,3] = torch.from_numpy(t).to(device).float()  # Add translation

    return transform


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    steps = [1_000, 10_000, 20_000, 30_000]
    max_steps = max(steps)
    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                normal_loss=True,
                dist_loss=True,
                save_ply=True,
                pose_opt=True,
                pose_opt_type="mlp",
                data_factor=1,
                max_steps=max_steps,
                eval_steps=steps,
                save_steps=steps,
                ply_steps=steps,
                enable_bg_model=True,
                enable_alpha_loss=True,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)

''' memo: splatfactow_model.py L1060~L1081
        use_cached_sh = False
        if camera.metadata is not None and "cam_idx" in camera.metadata:
            cam_idx = camera.metadata["cam_idx"]
            # 評価時のみキャッシュ判定
            # 複数のメトリクス(PSNR、SSIM、LPIPS)でレンダリングを繰り返さないように
            if self.last_cam_idx is not None and not self.training:
                use_cached_sh = cam_idx == self.last_cam_idx
                if cam_idx != self.last_cam_idx:
                    CONSOLE.log("Current camera idx is", cam_idx)
            self.last_cam_idx = cam_idx
            # indexを指定して取得する
            appearance_embed = self.appearance_embeds(
                torch.tensor(cam_idx, device=self.device)
            )
        else:
            if self.config.use_avg_appearance:
                # calculate the average appearance embedding
                appearance_embed = self.appearance_embeds.weight.mean(dim=0)
            else:
                appearance_embed = self.appearance_embeds(
                    torch.tensor(0, device=self.device)
                )

    appearance_embedをどの画像から作るか or エンベディングを平均したものを使うかをちゃんと考える必要があるかも

'''