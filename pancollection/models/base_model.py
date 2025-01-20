# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
import os

from udl_vis.Basis.python_sub_class import ModelDispatcher
import numpy as np
import torch
from torch.nn import functional as F
from pancollection.common.evaluate import analysis_accu, indexes_evaluation_FS
from pancollection.common.dataset import save_results
from udl_vis.Basis.dist_utils import get_dist_info
from scipy import io as sio


class PanSharpeningModel(ModelDispatcher, name="pansharpening"):

    _models = {}

    def __init__(self, device=None, model=None, criterion=None, logger=None):
        super(PanSharpeningModel, self).__init__()

        self.model = model
        self.criterion = criterion
        self.reg = False
        self.logger = logger
        self.device = device
        if model is not None:
            # define final_model_class with forward_task method to call different forward methods based on the specific task
            if hasattr(self.model, "module"):
                try:
                    self.model.module.forward_task = getattr(
                        self.model.module, f"forward_{self._name}"
                    )
                except:
                    self.model.module.forward_task = self.model.module.forward
            else:
                try:
                    self.model.forward_task = getattr(self.model, f"forward_{self._name}")
                except:
                    self.model.forward_task = self.model.forward
    
        if hasattr(self.model, "reg"):
            self.reg = self.model.reg

    def __init_subclass__(cls, name="", **kwargs):

        # print(name, cls)
        if name != "":
            cls._models[name] = cls
            cls._name = name
        else:
            cls._models[cls.__name__] = cls
            cls._name = cls.__name__
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')

    def l2_regularization(self, loss_dict, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in self.model.named_parameters():
            if "conv" in k and "weight" in k:
                penality = weight_decay * ((v.data**2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)
        if isinstance(loss_dict, dict):
            loss_dict["loss"] = loss_dict["loss"] + sum(regularizations)
            loss_dict["log_vars"].update(reg_loss=loss_dict["loss"])
        else:
            loss_dict = loss_dict + sum(regularizations)

        return loss_dict

    def train_step(self, data, **kwargs):
        """
        :param args:
                data = args[0]
                gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                                   data['ms'].cuda(), data['pan'].cuda()
        :param kwargs:
        :return:
        """
        log_vars = {}
        # data = {k: v.to(self.device) for k, v in batch.items()}

        # self.model.module.__class__.forward = self.model.module.__class__.train_step
        mode = kwargs.pop("mode")
        # sr = self.model(data, mode, **kwargs)  # module
        # loss = self.criterion(sr, data["gt"])
        # if self.reg:
        #     return self.l2_regularization(loss)
        # log_vars.update(loss=loss["loss"])
        # metrics = {"loss": loss["loss"], "log_vars": log_vars}
        log_vars = self.model(data, mode, **kwargs)
        if self.reg:
            return self.l2_regularization(log_vars["loss"])
        log_vars.update(loss=log_vars["loss"])
        metrics = {"loss": log_vars["loss"], "log_vars": log_vars}

        return metrics

    # @single_run
    def val_step(self, batch, **kwargs):
        # print(kwargs)
        data = {k: v.to(self.device) for k, v in batch.items()}
        gt = data.pop("gt")
        mode = kwargs.pop("mode")
        # self.model.module.__class__.forward = self.model.module.__class__.val_step
        sr = self.model(data, mode, **kwargs)  # module
        if not kwargs["test_mode"]:
            metrics = self.criterion(sr, gt)
        else:
            # sr: B, C, H, W
            # gt: B, C, H, W

            loss = self.criterion(sr, gt)["loss"]
            result_our = sr.permute(0, 2, 3, 1)
            gt = gt.permute(0, 2, 3, 1)
            if kwargs["test"] == "reduce":
                metrics = analysis_accu(gt, result_our, 4, prefix=mode)
                result_our = result_our * kwargs["img_range"]
                if kwargs["idx"] not in [220, 231, 236, 469, 766, 914]:
                    if kwargs["save_fmt"] is not None and os.path.isdir(
                        kwargs["save_dir"]
                    ):
                        save_results(
                            kwargs["idx"],
                            kwargs["save_dir"],
                            kwargs["filename"],
                            kwargs["save_fmt"],
                            result_our,
                            kwargs["img_range"],
                        )
                if kwargs["idx"] == 1257:
                    # oldPan
                    print(np.mean(self.SAM_list), np.mean(self.ERGAS_list))
            elif kwargs["test"] == "full":
                # TODO: batched evaluation
                metrics = indexes_evaluation_FS(
                    I_F=result_our.cpu().numpy(),
                    I_MS_LR=data["ms"].squeeze(0).permute(1, 2, 0).cpu().numpy(),
                    I_PAN=data["pan"].squeeze(0).permute(1, 2, 0).cpu().numpy(),
                    I_MS=data["lms"].squeeze(0).permute(1, 2, 0).cpu().numpy(),
                    L=11,
                    th_values=0,
                    sensor="wv3",
                    ratio=4,
                    Qblocks_size=32,
                    flagQNR=1,
                    prefix=mode,
                )
            metrics.update({f"{mode}_loss": loss})

        return {"log_vars": metrics}


if __name__ == "__main__":
    ...
