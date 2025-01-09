from udl_vis.Basis.option import Config
from pancollection import getDataSession, build_model
from pancollection.models.base_model import PanSharpeningModel
from udl_vis import trainer
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def hydra_run(config_path="configs", config_name="config", import_path=None):
    @hydra.main(config_path=config_path, config_name=config_name)
    def inner_func(cfg: DictConfig):
        if isinstance(cfg, DictConfig):
            cfg = Config(OmegaConf.to_container(cfg, resolve=True))
            cfg.merge_from_dict(cfg.args)
            cfg.__delattr__("args")
            hydra_cfg = HydraConfig.get()
            cfg.work_dir = hydra_cfg.runtime.output_dir
        print(cfg.pretty_text)
        cfg.backend = "lightning"
        if import_path is not None:
            cfg.import_path = import_path
        trainer.main(cfg, PanSharpeningModel, build_model, getDataSession)

    return inner_func()


if __name__ == "__main__":
    hydra_run()
