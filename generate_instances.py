import numpy as np

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
    return np.random.uniform(min_dim_bin, max_dim_bin, size=3)

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

if __name__ == "__main__":
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