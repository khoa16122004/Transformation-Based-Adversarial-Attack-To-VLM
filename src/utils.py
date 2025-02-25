import json
import random, os
import numpy as np
import torch
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def log(fitness_history, history, file_path):
    print()
    data = {
        "fitness_history": [fh.tolist() for fh in fitness_history],  # Chuyển np.array thành list
        "history": [h.tolist() for h in history]
    }
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=6)
            
        
        
    
    

    