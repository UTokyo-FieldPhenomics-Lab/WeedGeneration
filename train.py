import torch
from engine.trainer import trainer
import configuration

if configuration.model == "segan":
    from models import segan as model
elif configuration.model == "sngan":
    from models import segan as model
elif configuration.model == "wacgan_info":
    from models import wacgan_info as model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configuration.iterations = 100
# configuration.checkpoints_step_bin = 100
engine = trainer(configuration, model, device)
engine.train(ckpt=None)