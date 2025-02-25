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
        print("Min Fitness: ", np.min(fitness))
        
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx]
    best_value = fitness[best_idx]
    return best_solution, best_value, history, fitness_history


def PSO(func, bounds: List[Tuple[float, float]], pop_size: int, max_iter: int, w: float = 0.5, c1: float = 1.5, c2: float = 1.5):
    dim = len(bounds)
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim))
    vel = np.random.uniform(-1, 1, size=(pop_size, dim))
    
    fitness = func(pop)
    pbest = pop.copy() # population best
    pbest_fitness = fitness.copy()
    gbest_idx = np.argmin(fitness) 
    gbest = pop[gbest_idx] # partical best
    gbest_fitness = fitness[gbest_idx] # partical best fitness
    history = []
    fitness_history = []
    
    for _ in tqdm(range(max_iter)):
        r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
        vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop) # create new
        
        # new swarm
        pop = np.clip(pop + vel, [b[0] for b in bounds], [b[1] for b in bounds])
        new_fitness = func(pop)
        # update best swarm by the better canddidatess in new swarm
        mask = new_fitness < pbest_fitness
        mask = mask.reshape(-1,)
        pbest[mask] = pop[mask]
        pbest_fitness[mask] = new_fitness[mask]
        
        # update the best candidate
        if np.min(new_fitness) < gbest_fitness:
            gbest_idx = np.argmin(new_fitness)
            gbest = pop[gbest_idx]
            gbest_fitness = new_fitness[gbest_idx]
            
        history.append(pop.copy())
        fitness_history.append(gbest_fitness.copy())
        print("Min fitness: ", np.min(gbest_fitness))

        
    return gbest, gbest_fitness, history


