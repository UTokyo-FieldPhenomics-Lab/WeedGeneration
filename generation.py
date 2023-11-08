import torch

from engine import engine
import args

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    engine_raw = engine(args, device)

    engine_raw.generation(50, ckpt=f'./checkpoints/{engine_raw.config}/epoch1200.tar')