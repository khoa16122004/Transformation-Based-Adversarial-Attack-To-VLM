import argparse
import numpy as np
from utils import seed_everything
from models import OpenCLIP
from algorithm import DE_vectorize, PSO
import cv2
from fitness import Fitness
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Patch Manipulation")
    parser.add_argument('--pop_size', type=int, default=10, help="Population size")
    parser.add_argument('--max_iter', type=int, default=100, help="Number of iterations for the genetic algorithm")
    parser.add_argument('--seed', type=int, default=22520691)
    parser.add_argument('--model_name', type=str, default='ViT-H-14')
    parser.add_argument("--F", type=float, default=0.8)
    parser.add_argument("--CR", type=float, default=0.9)
    parser.add_argument("--algorithm", type=str, choices=["DE", "PSO"], default="DE")
    parser.add_argument("--annotation_file", type=str)
    parser.add_argument("--img_dir", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    success_rate = 0
    seed_everything(args.seed)
    
    
    with open(args.annotation_file, "r") as f:
        lines = [line.strip().split("\t") for line in f.readlines()]
    
    for i, [img_id, c_gt, c_tar] in enumerate(lines):
        img_path = os.path.join(args.img_dir, img_id)
        
        img = cv2.resize(cv2.imread(img_path), (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        vlm = OpenCLIP(args.model_name)
        fitness = Fitness(vlm,
                        img,
                        c_gt,
                        c_tar
                        )
        
        func = fitness.IG_IT_fitness
        bounds = [[0.8, 1.5], [0.8, 1.5], [-0.2, 0.2], [-0.2, 0.2], [-np.pi/5, np.pi/5], [0, 1], [0.5, 1.5]]
        if args.algorithm == 'DE':
            best_solution, best_value, history, fitness_history = DE_vectorize(func=fitness.IG_IT_fitness,
                                                                                bounds=bounds,
                                                                                pop_size=args.pop_size,
                                                                                F=args.F,
                                                                                CR=args.CR,
                                                                                max_iter=args.max_iter) 
        elif args.algorithm == "PSO":
            best_solution, best_value, history, fitness_history = PSO(func=fitness.IG_IT_fitness,
                                                                    bounds=bounds,
                                                                    pop_size=args.pop_size,
                                                                    max_iter=args.max_iter) 
        
        print("Best solution: ", best_solution)
        print("Best value: ", best_value)
        
        if best_value <= 0:
            success_rate += 1      
        break
    print("Success rate: ", success_rate / 100)  