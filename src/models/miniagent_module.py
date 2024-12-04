from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import BertModel


class MiniAgentModule(LightningModule):
    def __init__(
        self,
        bert_model: str,
        inst_proj_model: nn.Module,
        tool_proj_model: nn.Module,
        pred_model: nn.Module,
        lr: float,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False, ignore=["inst_proj_model", "tool_proj_model", "pred_model"]
        )

        self.bert_model = BertModel.from_pretrained(bert_model)

        self.inst_proj_model = inst_proj_model
        self.tool_proj_model = tool_proj_model
        self.pred_model = pred_model

        self.val_1_acc = Accuracy(task="binary")
        self.val_1_precision = MeanMetric()
        self.val_1_recall = MeanMetric()

        self.val_2_acc = Accuracy(task="binary")
        self.val_2_precision = MeanMetric()
        self.val_2_recall = MeanMetric()

        self.val_other_acc = Accuracy(task="binary")
        self.val_other_precision = MeanMetric()
        self.val_other_recall = MeanMetric()

        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def on_train_start(self) -> None:
        pass

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        B = batch["inst_ids"].shape[0]

        inst_ids = batch["inst_ids"]
        inst_mask = batch["inst_mask"]
        tool_ids = batch["tool_ids"]
        tool_mask = batch["tool_mask"]

        inst_z = self.bert_model(inst_ids, inst_mask, return_dict=False)[0]
        tool_z = self.bert_model(tool_ids, tool_mask, return_dict=False)[0]

        inst_emb = self.inst_proj_model(inst_z)
        tool_emb = self.tool_proj_model(tool_z)

        inst_emb_r = inst_emb.unsqueeze(1).repeat(1, B, 1).view(B * B, -1)
        tool_emb_r = tool_emb.unsqueeze(0).repeat(B, 1, 1).view(B * B, -1)

        pred = self.pred_model(inst_emb_r, tool_emb_r)  # [BxB, 1]
        pred = pred.view(B, B)  # [B, B]

        target = torch.eye(B, device=pred.device).float()

        pos_weight = torch.tensor([B - 1], device=pred.device)
        # pos_weight = torch.tensor([1], device=pred.device)
        loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

        self.log("train/loss", loss, on_step=True, sync_dist=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inst_ids = batch["inst_ids"]
        inst_mask = batch["inst_mask"]
        tool_ids = batch["tool_ids"]
        tool_mask = batch["tool_mask"]
        correct_tool_mask = batch["correct_tool_mask"]

        B = inst_ids.shape[0]  # batch size
        C = correct_tool_mask.shape[1]  # tool capacity
        tool_ids = tool_ids.view(-1, tool_ids.shape[-1])  # [B*C, L]
        tool_mask = tool_mask.view(-1, tool_mask.shape[-1])  # [B*C, L]

        inst_z = self.bert_model(inst_ids, inst_mask, return_dict=False)[0]
        tool_z = self.bert_model(tool_ids, tool_mask, return_dict=False)[0]

        inst_emb = self.inst_proj_model(inst_z)  # [B, D]
        tool_emb = self.tool_proj_model(tool_z)  # [B*C, D]

        inst_emb_r = inst_emb.unsqueeze(1).repeat(1, C, 1).view(B * C, -1)
        tool_emb_r = tool_emb.view(B * C, -1)

        pred = self.pred_model(inst_emb_r, tool_emb_r)  # [B*C, 1]
        pred = pred.view(B, C)
        pred = torch.sigmoid(pred)

        pred_tool_mask = pred > 0.5

        true_pos_mask = pred_tool_mask & correct_tool_mask

        one_tool_mask = correct_tool_mask.sum(dim=1) == 1
        two_tool_mask = correct_tool_mask.sum(dim=1) == 2
        other_mask = ~(one_tool_mask | two_tool_mask)

        # one tool
        one_tool_pos_sample = (
            (pred_tool_mask[one_tool_mask] == correct_tool_mask[one_tool_mask])
            .all(dim=1)
            .long()
        )

        one_tool_precision = true_pos_mask[one_tool_mask].sum(dim=1) / torch.clamp(
            pred_tool_mask[one_tool_mask].sum(dim=1), min=1
        )

        one_tool_recall = true_pos_mask[one_tool_mask].sum(dim=1) / torch.clamp(
            correct_tool_mask[one_tool_mask].sum(dim=1), min=1
        )

        # two tool
        two_tool_pos_sample = (
            (pred_tool_mask[two_tool_mask] == correct_tool_mask[two_tool_mask])
            .all(dim=1)
            .long()
        )

        two_tool_precision = true_pos_mask[two_tool_mask].sum(dim=1) / torch.clamp(
            pred_tool_mask[two_tool_mask].sum(dim=1), min=1
        )

        two_tool_recall = true_pos_mask[two_tool_mask].sum(dim=1) / torch.clamp(
            correct_tool_mask[two_tool_mask].sum(dim=1), min=1
        )

        # other
        other_pos_sample = (
            (pred_tool_mask[other_mask] == correct_tool_mask[other_mask])
            .all(dim=1)
            .long()
        )

        other_precision = true_pos_mask[other_mask].sum(dim=1) / torch.clamp(
            pred_tool_mask[other_mask].sum(dim=1), min=1
        )

        other_recall = true_pos_mask[other_mask].sum(dim=1) / torch.clamp(
            correct_tool_mask[other_mask].sum(dim=1), min=1
        )

        if one_tool_pos_sample.sum().item() > 0:
            self.val_1_acc.update(
                one_tool_pos_sample, torch.ones_like(one_tool_pos_sample)
            )
            self.val_1_precision.update(one_tool_precision)
            self.val_1_recall.update(one_tool_recall)

            self.log(
                "val/1_acc",
                self.val_1_acc,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            self.log(
                "val/1_precision",
                self.val_1_precision,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            self.log(
                "val/1_recall",
                self.val_1_recall,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

        if two_tool_pos_sample.sum().item() > 0:
            self.val_2_acc.update(
                two_tool_pos_sample, torch.ones_like(two_tool_pos_sample)
            )
            self.val_2_precision.update(two_tool_precision)
            self.val_2_recall.update(two_tool_recall)

            self.log(
                "val/2_acc",
                self.val_2_acc,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            self.log(
                "val/2_precision",
                self.val_2_precision,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            self.log(
                "val/2_recall",
                self.val_2_recall,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

        if other_pos_sample.sum().item() > 0:
            self.val_other_acc.update(
                other_pos_sample, torch.ones_like(other_pos_sample)
            )
            self.val_other_precision.update(other_precision)
            self.val_other_recall.update(other_recall)

            self.log(
                "val/other_acc",
                self.val_other_acc,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            self.log(
                "val/other_precision",
                self.val_other_precision,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            self.log(
                "val/other_recall",
                self.val_other_recall,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {"params": self.bert_model.parameters(), "lr": 1e-5},
                {
                    "params": list(self.inst_proj_model.parameters())
                    + list(self.tool_proj_model.parameters())
                    + list(self.pred_model.parameters()),
                    "lr": self.lr,
                },
            ],
            weight_decay=1e-4,
        )
        return opt
