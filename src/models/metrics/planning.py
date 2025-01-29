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

    def __init__(self, nav_with_route, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nav_with_route = nav_with_route

        self.ego_loss_prefix = f"{self.prefix}/ego"
        del kwargs["prefix"]

        self.route_loss = RouteLoss(prefix=self.ego_loss_prefix, **kwargs)
        self.goal_loss = GoalLoss(prefix=self.ego_loss_prefix, **kwargs)
        self.gt_loss = NllMetrics(prefix=self.ego_loss_prefix, **kwargs)

        self.add_state("ego_gt_loss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ego_route_loss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ego_goal_loss", default=torch.tensor(0), dist_reduce_fx="sum")

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
        gt_route_goal: Tensor,
        gt_route_goal_valid: Tensor,
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
            gt_route_goal: [n_scene, n_agent, n_pl_route, 2]
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
        ego_batch["gt_route_goal"] = gt_route_goal[ego_mask[0]].view(n_scene, 1, 2)
        ego_batch["gt_route_goal_valid"] = gt_route_goal_valid[ego_mask[0]].view(
            n_scene, 1, 1
        )

        # ! loss
        ego_nll_dict = self.gt_loss.forward(**ego_batch)
        self.ego_gt_loss = ego_nll_dict[f"{self.ego_loss_prefix}/loss"]
        if self.nav_with_route:
            ego_route_loss_dict = self.route_loss.forward(**ego_batch)
            self.ego_route_loss = ego_route_loss_dict[f"{self.ego_loss_prefix}/loss"]
        else:
            ego_goal_loss_dict = self.goal_loss.forward(**ego_batch)
            self.ego_goal_loss = ego_goal_loss_dict[f"{self.ego_loss_prefix}/loss"]

    def forward(self, **kwargs) -> Dict[str, Tensor]:
        loss_dict = super().forward(**kwargs)

        self.planning_metrics(**kwargs)

        loss_dict[f"{self.ego_loss_prefix}/gt_loss"] = self.ego_gt_loss
        if self.nav_with_route:
            loss_dict[f"{self.ego_loss_prefix}/route_loss"] = self.ego_route_loss
            ego_nav_loss = self.ego_route_loss
        else:
            loss_dict[f"{self.ego_loss_prefix}/goal_loss"] = self.ego_goal_loss
            ego_nav_loss = self.ego_goal_loss

        ego_planning_loss = self.ego_gt_loss + ego_nav_loss / ego_nav_loss.detach()
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


class GoalLoss(Metric):
    def __init__(self, prefix: str, n_decoders: int, **kwargs) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.n_decoders = n_decoders

        self.mse_loss = torch.nn.MSELoss()
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_conf: Tensor,
        pred_pos: Tensor,
        gt_valid: Tensor,
        gt_route_goal_valid: Tensor,
        gt_route_goal: Tensor,
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

        # ! loss for all modes simultaneously, since all ego predictions should be aligned with the goal
        # Calculate loss for each decoder and scene separately, because the number of valid
        # prediction points can differ between scenes.
        for i in range(n_decoder):
            for j in range(n_scene):
                # (1 2)
                gt_route_goal_scene = gt_route_goal[j]

                # Find the last valid index along n_step_future
                # Using torch.where to find valid indices, then taking the max index per [n_agent, n_pred]
                last_valid_indices = torch.argmax(
                    avails.int()[i, j].flip(dims=[-1]), dim=-1
                )
                last_valid_indices = (
                    n_step_future - 1 - last_valid_indices
                )  # Convert flipped indices to original indices

                # Use the last valid indices to index into the data tensor
                # Gather indices for advanced indexing
                agent_indices = (
                    torch.arange(n_agent).unsqueeze(1).expand(n_agent, n_pred)
                )
                pred_indices = torch.arange(n_pred).unsqueeze(0).expand(n_agent, n_pred)
                final_pred_pos = pred_pos[
                    i, j, agent_indices, pred_indices, last_valid_indices, :
                ]

                self.loss += self.mse_loss(
                    final_pred_pos,
                    gt_route_goal_scene.unsqueeze(1).expand(n_agent, n_pred, 2),
                )

    def compute(self) -> Dict[str, Tensor]:
        # Compute the MSE loss across all samples
        mse_loss = self.loss / (self.n_decoders * self.n_scene)

        # Return the loss as a dictionary
        out_dict = {f"{self.prefix}/loss": mse_loss}
        return out_dict
