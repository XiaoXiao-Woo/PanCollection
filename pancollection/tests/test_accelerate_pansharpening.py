from udl_vis import trainer
from udl_vis.Basis.option import Config
from pancollection import getDataSession
from pancollection.models.FusionNet.model_fusionnet import build_fusionnet
from pancollection.models.base_model import PanSharpeningModel
from omegaconf import OmegaConf, DictConfig
import hydra
import os
from hydra.core.hydra_config import HydraConfig

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.get("MASTER_ADDR", "localhost")
os.environ.get("MASTER_PORT", "24567")

@hydra.main(config_path="/home/dsq/NIPS/PanCollection/pancollection/configs/", 
            config_name="model")
def hydra_run(cfg: DictConfig):
    if isinstance(cfg, DictConfig):
        cfg = Config(OmegaConf.to_container(cfg, resolve=True))
        cfg.merge_from_dict(cfg.args)
        cfg.__delattr__("args")
        hydra_cfg = HydraConfig.get()
        cfg.work_dir = hydra_cfg.runtime.output_dir
    print(cfg.pretty_text)
    cfg.backend = "accelerate"
    cfg.launcher = "accelerate"
    cfg.dataset_type = "Dummy"
    
    trainer.main(cfg, PanSharpeningModel, build_fusionnet, getDataSession)


if __name__ == "__main__":
    hydra_run()
