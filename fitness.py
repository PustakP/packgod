import numpy as np
import random
from typing import List, Tuple

MIN_DIM_BIN = 10
MAX_DIM_BIN = 20
MIN_SIZE_OBJ = 1
NUM_OBJECTS = 45
MAX_ASPECT_RATIO = 5

class Bin:
    def __init__(self, dim):
        self.dim = np.array(dim)

    def __repr__(self):
        length, width, height = self.dim
        return f"Bin({length:.2f} x {width:.2f} x {height:.2f})"

    def volume(self):
        return np.prod(self.dim)

class Box:
    def __init__(self, id, dim):
        self.id = id
        self.dim = np.array(dim)

    def __repr__(self):
        length, width, height = self.dim
        return f"Box-{self.id}({length:.2f} x {width:.2f} x {height:.2f})"

    def volume(self):
        return np.prod(self.dim)

def generate_bin(min_dim_bin, max_dim_bin):
    """
    generate a container within a range
    :param min_dim_bin: float, min value for each dimension
    :param max_dim_bin: float, max value for each dimension
    :return: numpy array of floats (length, width, height)
    """
    return np.random.uniform(min_dim_bin, max_dim_bin + 1, size=3)

def is_valid_aspect_ratio(dim, max_ratio=MAX_ASPECT_RATIO):
    """
    check if the aspect ratio is within an acceptable range
    :param dim: array, dimensions of the box
    :param max_ratio: float, maximum allowed ratio between largest and smallest dim
    :return: bool
    """
    ratios = dim / dim.min()
    return ratios.max() <= max_ratio

def generate_objects(num_objects, bin: Bin, min_size_obj):
    """
    generate a list of boxes with random dimensions that:
    - individually fit within the bin
    - have a controlled aspect ratio
    - collectively do not exceed an estimated fill ratio
    :param num_objects: int
    :param bin: Bin instance
    :param min_size_obj: float
    :return: list of Box instances
    """
    bin_dim = bin.dim
    bin_volume = bin.volume()

    max_fill_ratio = min(0.95, 0.5 + 0.5 * num_objects / 30) #can be tuned for more realistic packing density
    target_total_volume = bin_volume * max_fill_ratio
    target_avg_volume = target_total_volume / num_objects

    objects = []

    for obj_id in range(1, num_objects + 1):
        for i in range(10):
            dims = np.random.uniform(min_size_obj, bin_dim, size=3)
            vol = np.prod(dims)

            if vol <= target_avg_volume * 2 and is_valid_aspect_ratio(dims):
                objects.append(Box(obj_id, dims))
                break
        else:
            scale = (target_avg_volume / vol) ** (1/3)
            dims = dims * scale
            objects.append(Box(obj_id, dims))

    return objects

def calculate_normalized_fitness(bin: Bin, boxes: List[Box], positions: List[Tuple[float, float, float]]) -> float:
    """
    Returns a fitness score between -1.0 (worst) and 1.0 (best).
    1.0 = All boxes placed perfectly with no collisions and max space used
    0.0 = No boxes placed
    -1.0 = Maximum violations (all boxes out of bounds or colliding)
    """
    total_possible_volume = sum(box.volume() for box in boxes)
    if total_possible_volume == 0:
        return 0.0  # Edge case: no boxes to place

    used_volume = 0.0
    violations = 0
    max_violations = len(boxes) * 2  # Worst case: all boxes out of bounds AND colliding

    for i, (box, (x, y, z)) in enumerate(zip(boxes, positions)):
        # Check boundary violations
        boundary_ok = True
        if (x < 0 or y < 0 or z < 0 or
            x + box.dim[0] > bin.dim[0] or
            y + box.dim[1] > bin.dim[1] or
            z + box.dim[2] > bin.dim[2]):
            violations += 1
            boundary_ok = False

        # Check collisions if within boundaries
        if boundary_ok:
            used_volume += box.volume()
            for j in range(i + 1, len(boxes)):
                other_box, (ox, oy, oz) = boxes[j], positions[j]
                if (x < ox + other_box.dim[0] and
                    x + box.dim[0] > ox and
                    y < oy + other_box.dim[1] and
                    y + box.dim[1] > oy and
                    z < oz + other_box.dim[2] and
                    z + box.dim[2] > oz):
                    violations += 1

    # Normalize volume usage (0 to 1)
    volume_score = used_volume / total_possible_volume

    # Normalize violations (0 to 1)
    violation_score = 1 - (violations / max_violations) if max_violations > 0 else 1

    # Combined score (weighted average)
    fitness = (0.7 * volume_score) + (0.3 * violation_score)  # 70% volume, 30% legality

    return max(0.0, min(1.0, fitness))  # Clamp range of fitness score to [0, 1]

if __name__ == "__main__":
    # Initialize bin and boxes
    bin_dim = generate_bin(MIN_DIM_BIN, MAX_DIM_BIN)
    bin = Bin(bin_dim)
    boxes = generate_objects(NUM_OBJECTS, bin, MIN_SIZE_OBJ)

    print("Bin:")
    print(bin)
    print("\nBoxes:")
    for box in boxes:
        print(box)

    total_obj_volume = sum(box.volume() for box in boxes)
    print(f"\nTotal box volume: {total_obj_volume:.2f}")
    print(f"Bin volume: {bin.volume():.2f}")
    print(f"Fill ratio: {total_obj_volume / bin.volume():.2f}")

    # Test cases
    test_positions = [
        # Case 1: All boxes perfectly placed (should be ~1.0)
        [(i * 1.1, 0, 0) for i, box in enumerate(boxes)],
        
        # Case 2: All boxes at origin (should be low)
        [(0, 0, 0)] * len(boxes),
        
        # Case 3: Random placement
        [(random.uniform(0, bin.dim[0] - box.dim[0]), 
          random.uniform(0, bin.dim[1] - box.dim[1]), 
          random.uniform(0, bin.dim[2] - box.dim[2])) for box in boxes],
        
        # Case 4: All boxes out of bounds (should be 0.0)
        [(-1, -1, -1)] * len(boxes)
    ]

    for i, positions in enumerate(test_positions):
        fitness = calculate_normalized_fitness(bin, boxes, positions)
        print(f"\nTest Case {i+1} Fitness: {fitness:.4f}")
        if i == 0:
            print("(Expected: Close to 1.0 - perfect packing)")
        elif i == 1:
            print("(Expected: Low - all boxes colliding)")
        elif i == 3:
            print("(Expected: 0.0 - all boxes out of bounds)")