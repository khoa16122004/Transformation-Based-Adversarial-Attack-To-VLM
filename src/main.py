import argparse
from individual import Flow
from algorithm import GA
from fitness import Fitness
import numpy as np
import os
from torchvision import transforms
import random
from torchvision.utils import save_image
import pickle as pkl
from utils import seed_everything
from PIL import Image
from model import OpenCLIP
from algorithm import DE_vectorize
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Patch Manipulation")
    parser.add_argument('--pop_size', type=int, default=10, help="Population size")
    parser.add_argument('--n_iters', type=int, default=100, help="Number of iterations for the genetic algorithm")
    parser.add_argument('--seed', type=int, default=22520691)
    parser.add_argument('--model_name', type=str, default='ViT-H-14')
    parser.add_argument('--fitness_type', type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    seed_everything(args.seed)
    
    img_path = r"images/lionsea.jpg"
    img = cv2.resize(cv2.imread('images/dog.jpg'), (224, 224))
    c_gt = "A proud sea lion basking in the sun on the rocks, with a seabird nearby."
    
    vlm = OpenCLIP(args.model_name)
    fitness = Fitness(vlm,
                      img,
                      c_gt,
                      )
    
    func = fitness.IG_IT_fitness
    
    # [[0.8, 1.5], [0.8, 1.5], [-0.2, 0.2], [-0.2, 0.2], [-np.pi/6, np.pi/6]]
    bounds = [[0.8, 1.5], [0.8, 1.5], [-0.2, 0.2], [-0.2, 0.2], [-np.pi/6, np.pi/6]]
    
    best_solution, best_value, history, fitness_history = DE_vectorize(img,
                                                      func=func, bounds=bounds,
                                                      pop_size=args.pop_size,
                                                      F=args.F,
                                                      CR=args.CR,
                                                      max_iter=args.max_iter)    

    