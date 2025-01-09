from typing import Dict, Optional
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from pytorch3d.loss import chamfer_distance

from models.metrics.nll import NllMetrics


class EgoPlanningMetrics(NllMetrics):
    """
    This Loss build upon the NllMetrics class and extends it with additional metrics for ego planning.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight_ego_pos = 1.0
        self.weight_ego_route = 1.0

        self.ego_loss_prefix = f"{self.prefix}/ego"
        del kwargs["prefix"]

        self.route_loss = RouteLoss(prefix=self.ego_loss_prefix, **kwargs)
        self.gt_loss = NllMetrics(prefix=self.ego_loss_prefix, **kwargs)

        self.add_state("ego_gt_loss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ego_route_loss", default=torch.tensor(0), dist_reduce_fx="sum")

    def planning_metrics(
        self,
        pred_valid: Tensor,
        pred_conf: Tensor,
        pred_pos: Tensor,
        pred_cov: Optional[Tensor],
        ref_role: Tensor,
        ref_type: Tensor,
        gt_valid: Tensor,
        gt_pos: Tensor,
        gt_spd: Tensor,
        gt_vel: Tensor,
        gt_yaw_bbox: Tensor,
        gt_cmd: Tensor,
        gt_route_valid: Tensor,
        gt_route_pos: Tensor,
        **kwargs,
    ) -> None:
        """
        Args:
            pred_valid: [n_scene, n_agent], bool, n_agent = n_target
            pred_conf: [n_decoder, n_scene, n_agent, n_pred], not normalized!
            pred_pos: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2]
            pred_cov: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2, 2]
            ref_role: [n_scene, n_agent, 3], one hot bool [sdc=0, interest=1, predict=2]
            ref_type: [n_scene, n_agent, 3], one hot bool [veh=0, ped=1, cyc=2]
            gt_valid: [n_scene, n_agent, n_step_future], bool
            gt_pos: [n_scene, n_agent, n_step_future, 2]
            gt_spd: [n_scene, n_agent, n_step_future, 1]
            gt_vel: [n_scene, n_agent, n_step_future, 2]
            gt_yaw_bbox: [n_scene, n_agent, n_step_future, 1]
            gt_cmd: [n_scene, n_agent, 8], one hot bool
            gt_route_valid: [n_scene, n_agent, n_pl_route, n_pl_node], bool
            gt_route_pos: [n_scene, n_agent, n_pl_route, n_pl_node, 2]
        """
        n_decoder, n_scene, n_agent, n_pred, n_step_future, _ = pred_pos.shape
        _, _, n_route, n_pl_node = gt_route_valid.shape

        # ! create ego mask
        ego_mask = ref_role[..., 0] == True
        ego_mask = ego_mask.expand(n_decoder, -1, -1)
        assert (
            ego_mask.sum() / (n_decoder * n_scene) == 1
        ), "Only one ego agent supported"

        ego_batch = {}
        # ! keep ego data; delete all other agents
        ego_batch["pred_valid"] = pred_valid[ego_mask[0]].view(n_scene, 1)
        ego_batch["pred_conf"] = pred_conf[ego_mask].view(n_decoder, n_scene, 1, n_pred)
        ego_batch["pred_pos"] = pred_pos[ego_mask].view(
            n_decoder, n_scene, 1, n_pred, n_step_future, 2
        )
        ego_batch["pred_spd"] = None
        ego_batch["pred_vel"] = None
        ego_batch["pred_yaw_bbox"] = None
        ego_batch["pred_cov"] = pred_cov[ego_mask].view(
            n_decoder, n_scene, 1, n_pred, n_step_future, 2, 2
        )
        ego_batch["ref_role"] = ref_role[ego_mask[0]].view(n_scene, 1, 3)
        ego_batch["ref_type"] = ref_type[ego_mask[0]].view(n_scene, 1, 3)
        ego_batch["gt_valid"] = gt_valid[ego_mask[0]].view(n_scene, 1, n_step_future)
        ego_batch["gt_pos"] = gt_pos[ego_mask[0]].view(n_scene, 1, n_step_future, 2)
        ego_batch["gt_spd"] = gt_spd[ego_mask[0]].view(n_scene, 1, n_step_future, 1)
        ego_batch["gt_vel"] = gt_vel[ego_mask[0]].view(n_scene, 1, n_step_future, 2)
        ego_batch["gt_yaw_bbox"] = gt_yaw_bbox[ego_mask[0]].view(
            n_scene, 1, n_step_future, 1
        )
        ego_batch["gt_cmd"] = gt_cmd[ego_mask[0]].view(n_scene, 1, 8)
        ego_batch["gt_route_valid"] = gt_route_valid[ego_mask[0]].view(
            n_scene, 1, n_route, n_pl_node
        )
        ego_batch["gt_route_pos"] = gt_route_pos[ego_mask[0]].view(
            n_scene, 1, n_route, n_pl_node, 2
        )

        # ! loss
        ego_nll_dict = self.gt_loss.forward(**ego_batch)
        ego_route_loss_dict = self.route_loss.forward(**ego_batch)

        self.ego_gt_loss = ego_nll_dict[f"{self.ego_loss_prefix}/loss"]
        self.ego_route_loss = ego_route_loss_dict[f"{self.ego_loss_prefix}/loss"]

    def planning_loss(
        self,
        gt_loss: Tensor,
        route_loss: Tensor,
        gt_weight: float,
        route_weight: float,
        norm_weights=True,
        reduction: Optional[str] = "sum",
    ) -> Tensor:
        """
        Args:
            gt_loss: [1], float
            route_loss: [1], float
            gt_weight: float
            route_weight: float
            norm_weights: bool, whether to normalize the weights
            reduction: str, "mean" or "sum"
        """
        if norm_weights:
            gt_weight /= gt_weight + route_weight
            route_weight /= gt_weight + route_weight
        if reduction == "mean":
            return torch.mean(
                torch.tensor([gt_weight, route_weight])
                * torch.tensor([gt_loss, route_loss])
            )
        elif reduction == "sum":
            return gt_weight * gt_loss + route_weight * route_loss
        else:
            raise ValueError(f"reduction {reduction} not supported")

    def forward(self, **kwargs) -> Dict[str, Tensor]:
        loss_dict = super().forward(**kwargs)
        self.planning_metrics(**kwargs)
        loss_dict[f"{self.ego_loss_prefix}/gt_loss"] = self.ego_gt_loss
        loss_dict[f"{self.ego_loss_prefix}/route_loss"] = self.ego_route_loss
        ego_planning_loss = self.planning_loss(
            self.ego_gt_loss,
            self.ego_route_loss,
            self.weight_ego_pos,
            self.weight_ego_route,
        )
        loss_dict[f"{self.ego_loss_prefix}/loss"] = ego_planning_loss
        return loss_dict


class RouteLoss(Metric):
    def __init__(self, prefix: str, n_decoders: int, **kwargs) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.n_decoders = n_decoders

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_conf: Tensor,
        pred_pos: Tensor,
        gt_valid: Tensor,
        gt_route_valid: Tensor,
        gt_route_pos: Tensor,
        **kwargs,
    ) -> None:
        n_decoder, n_scene, n_agent, n_pred, n_step_future, _ = pred_pos.shape
        self.n_scene = n_scene
        assert n_agent == 1, "Only one ego agent supported"

        # ! prepare avails
        # [n_scene, n_agent, n_step_future]
        avails = gt_valid
        # [n_decoder, n_scene, n_agent, n_step_future]
        avails = avails.unsqueeze(0).expand(n_decoder, -1, -1, -1)
        if n_decoder > 1:
            # [n_decoder], randomly train ensembles with 50% of chance
            mask_ensemble = torch.bernoulli(
                0.5 * torch.ones_like(pred_conf[:, 0, 0, 0])
            ).bool()
            # make sure at least one ensemble is trained
            if not mask_ensemble.any():
                mask_ensemble[torch.randint(0, n_decoder, (1,))] |= True
            avails = avails & mask_ensemble[:, None, None, None]
        # [n_decoder, n_scene, n_agent, n_pred, n_step_future]
        avails = avails.unsqueeze(3).expand(-1, -1, -1, n_pred, -1)

        # ! loss for all modes simultaneously, since all ego predictions should be on-route
        # Calculate loss for each decoder and scene separately, because the number of valid
        # route and prediction points can differ between scenes.
        for i in range(n_decoder):
            for j in range(n_scene):
                # ((n_agent n_route n_pl_node_modified) 2)
                gt_route_pos_i_j = gt_route_pos[j][gt_route_valid[j]].unsqueeze(0)
                # ((n_agent n_pred n_step_future_modified) 2)
                pred_pos_i_j = pred_pos[i, j][avails[i, j]].unsqueeze(0)

                # Compute Chamfer distance between the batch of predicted positions and route positions
                chamfer_loss, _ = chamfer_distance(
                    pred_pos_i_j, gt_route_pos_i_j, single_directional=True
                )
                self.loss += chamfer_loss

    def compute(self) -> Dict[str, Tensor]:
        # Compute the average Chamfer loss across all samples
        avg_chamfer_loss = self.loss / (self.n_decoders * self.n_scene)

        # Return the loss as a dictionary
        out_dict = {f"{self.prefix}/loss": avg_chamfer_loss}
        return out_dict
