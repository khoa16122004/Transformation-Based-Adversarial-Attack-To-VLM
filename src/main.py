import argparse
import numpy as np
from utils import seed_everything
from models import OpenCLIP
from algorithm import DE_vectorize
import cv2
from fitness import Fitness

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Patch Manipulation")
    parser.add_argument('--pop_size', type=int, default=10, help="Population size")
    parser.add_argument('--max_iter', type=int, default=100, help="Number of iterations for the genetic algorithm")
    parser.add_argument('--seed', type=int, default=22520691)
    parser.add_argument('--model_name', type=str, default='ViT-H-14')
    parser.add_argument("--F", type=float, default=0.8)
    parser.add_argument("--CR", type=float, default=0.9)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    seed_everything(args.seed)
    
    img_path = r"images/lionsea.jpg"
    img = cv2.resize(cv2.imread('images/lionsea.jpg'), (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    c_gt = "A proud sea lion basking in the sun on the rocks."
    c_tar = "A proud dog basking in the sun on the rocks."
    vlm = OpenCLIP(args.model_name)
    fitness = Fitness(vlm,
                      img,
                      c_gt,
                      c_tar
                      )
    
    func = fitness.IG_IT_fitness
    
    # [[0.8, 1.5], [0.8, 1.5], [-0.2, 0.2], [-0.2, 0.2], [-np.pi/6, np.pi/6]]
    bounds = [[0.8, 1.5], [0.8, 1.5], [-0.3, 0.3], [-0.3, 0.3], [-np.pi/5, np.pi/5], [0, 1], [0.5, 1.5]]
    
    best_solution, best_value, history, fitness_history = DE_vectorize(func=fitness.IG_IT_fitness,
                                                                       bounds=bounds,
                                                                       pop_size=args.pop_size,
                                                                       F=args.F,
                                                                       CR=args.CR,
                                                                       max_iter=args.max_iter) 
    
    print("Best solution: ", best_solution)
    print("Best value: ", best_value)

    