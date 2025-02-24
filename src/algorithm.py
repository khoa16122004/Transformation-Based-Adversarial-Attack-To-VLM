import numpy as np
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm

def DE_vectorize(func, bounds: List[Tuple[float, float]], pop_size: int, F: float, CR: float, max_iter: int):
    dim = len(bounds)
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim))
    
    fitness = func(pop)
    history = []
    fitness_history = []
    
    for _ in tqdm(range(max_iter)):
        r1, r2, r3 = [], [], []
        for i in range(pop_size):
            r1_, r2_, r3_ = np.random.choice([idx for idx in range(pop_size) if idx != i], size=3, replace=False) # not duplicated sample
            r1.append(r1_)
            r2.append(r2_)
            r3.append(r3_)
        
        x1, x2, x3 = pop[r1], pop[r2], pop[r3]
        v = x1 + F * (x2 - x3) # pop_size x D
        v = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds]) # pop_size x D
        
        j_rand = np.random.randint(0, dim, size=pop_size)
        mask = np.random.rand(pop_size, dim) < CR
        mask[np.arange(pop_size), j_rand] = True
        u = np.where(mask, v, pop) # pop_size x D
        
        new_fitness = func(u)
        improved = new_fitness < fitness
        pop[improved] = u[improved]
        fitness[improved] = new_fitness[improved]
        history.append(pop.copy())
        fitness_history.append(fitness.copy())
        
        
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx]
    best_value = fitness[best_idx]
    return best_solution, best_value, history, fitness_history


