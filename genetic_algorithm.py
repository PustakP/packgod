import numpy as np
import random
from typing import List, Tuple, Optional
from generate_instances import Bin, Box, generate_bin, generate_objects, MIN_DIM_BIN, MAX_DIM_BIN, MIN_SIZE_OBJ, NUM_OBJECTS
from crossover import crossover, repair, Chromosome, UNPLACED
from fitness import calculate_normalized_fitness

class GeneticAlgorithm:
    def __init__(self, bin_inst: Bin, boxes: List[Box], pop_size=50, mut_rate=0.1, elite_size=5):
        self.bin_inst = bin_inst
        self.boxes = boxes
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.elite_size = elite_size
        self.population: List[Chromosome] = []
        self.fitness_history = []
        self.best_solutions = []
        
    def init_pop(self) -> List[Chromosome]:
        """init random pop"""
        pop = []
        for _ in range(self.pop_size):
            chrom = []
            for box in self.boxes:
                if random.random() < 0.7:  # 70% chance to place
                    pos = [
                        random.uniform(0, max(0.1, self.bin_inst.dim[0] - box.dim[0])),
                        random.uniform(0, max(0.1, self.bin_inst.dim[1] - box.dim[1])),
                        random.uniform(0, max(0.1, self.bin_inst.dim[2] - box.dim[2]))
                    ]
                    chrom.append([1.0] + pos)
                else:
                    chrom.append(UNPLACED)
            # repair invalid placements
            chrom = repair(chrom, self.boxes, self.bin_inst)
            pop.append(chrom)
        return pop
    
    def calc_fitness(self, chrom: Chromosome) -> float:
        """calc fitness for chromosome"""
        positions = []
        for gene in chrom:
            if gene[0] > 0:  # placed
                positions.append((gene[1], gene[2], gene[3]))
            else:
                positions.append((0, 0, 0))  # dummy pos for unplaced
        return calculate_normalized_fitness(self.bin_inst, self.boxes, positions)
    
    def select_parents(self, pop: List[Chromosome], fitness_scores: List[float]) -> Tuple[Chromosome, Chromosome]:
        """tournament selection"""
        def tournament(k=3):
            candidates = random.sample(list(zip(pop, fitness_scores)), k)
            return max(candidates, key=lambda x: x[1])[0]
        return tournament(), tournament()
    
    def mutate(self, chrom: Chromosome) -> Chromosome:
        """mutate chromosome"""
        mutated = []
        for i, gene in enumerate(chrom):
            if random.random() < self.mut_rate:
                box = self.boxes[i]
                if random.random() < 0.5 and gene[0] > 0:  # modify existing placement
                    new_pos = [
                        gene[1] + random.gauss(0, 1),
                        gene[2] + random.gauss(0, 1), 
                        gene[3] + random.gauss(0, 1)
                    ]
                    mutated.append([1.0] + new_pos)
                elif random.random() < 0.3:  # toggle placement
                    if gene[0] > 0:
                        mutated.append(UNPLACED)
                    else:
                        pos = [
                            random.uniform(0, max(0.1, self.bin_inst.dim[0] - box.dim[0])),
                            random.uniform(0, max(0.1, self.bin_inst.dim[1] - box.dim[1])),
                            random.uniform(0, max(0.1, self.bin_inst.dim[2] - box.dim[2]))
                        ]
                        mutated.append([1.0] + pos)
                else:
                    mutated.append(gene[:])
            else:
                mutated.append(gene[:])
        return repair(mutated, self.boxes, self.bin_inst)
    
    def evolve(self, generations=100):
        """run ga for specified generations"""
        self.population = self.init_pop()
        
        for gen in range(generations):
            # calc fitness for all
            fitness_scores = [self.calc_fitness(chrom) for chrom in self.population]
            
            # track best
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_chrom = self.population[best_idx]
            
            self.fitness_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'worst_fitness': np.min(fitness_scores)
            })
            self.best_solutions.append(best_chrom[:])
            
            if gen % 10 == 0:
                print(f"gen {gen}: best={best_fitness:.4f}, avg={np.mean(fitness_scores):.4f}")
            
            # create new pop
            new_pop = []
            
            # elitism - keep best
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_pop.append(self.population[idx][:])
            
            # generate offspring
            while len(new_pop) < self.pop_size:
                p1, p2 = self.select_parents(self.population, fitness_scores)
                child = crossover(p1, p2, self.boxes, self.bin_inst)
                child = self.mutate(child)
                new_pop.append(child)
            
            self.population = new_pop
        
        return self.population[np.argmax([self.calc_fitness(c) for c in self.population])]

def run_ga():
    """main runner function"""
    # gen problem instance
    bin_dim = generate_bin(MIN_DIM_BIN, MAX_DIM_BIN)
    bin_inst = Bin(bin_dim)
    boxes = generate_objects(NUM_OBJECTS, bin_inst, MIN_SIZE_OBJ)
    
    print(f"bin: {bin_inst}")
    print(f"boxes: {len(boxes)}")
    print(f"total box vol: {sum(b.volume() for b in boxes):.2f}")
    print(f"bin vol: {bin_inst.volume():.2f}")
    print(f"fill ratio: {sum(b.volume() for b in boxes) / bin_inst.volume():.2f}\n")
    
    # run ga
    ga = GeneticAlgorithm(bin_inst, boxes, pop_size=50, mut_rate=0.15, elite_size=5)
    best_solution = ga.evolve(generations=100)
    
    # final results
    final_fitness = ga.calc_fitness(best_solution)
    print(f"\nfinal best fitness: {final_fitness:.4f}")
    
    placed_count = sum(1 for gene in best_solution if gene[0] > 0)
    print(f"boxes placed: {placed_count}/{len(boxes)}")
    
    return ga, best_solution

if __name__ == "__main__":
    ga, solution = run_ga()
