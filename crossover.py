from typing import List, Tuple, Optional
import numpy as np

Gene = List[float]
Chromosome = List[Gene]
PlacedBox = Tuple[np.ndarray, np.ndarray]
UNPLACED: Gene = [0.0, 0.0, 0.0, 0.0]


def is_valid_pos(pos: np.ndarray, box_dim: np.ndarray, bin_dim: np.ndarray) -> bool:
    """Check if the box fits inside the bin at the given position."""
    return np.all(pos >= 0) and np.all(pos + box_dim <= bin_dim)


def check_overlap(pos: np.ndarray, box_dim: np.ndarray, placed: List[PlacedBox]) -> bool:
    """Check if the box at `pos` overlaps with any placed boxes."""
    for placed_pos, placed_dim in placed:
        if not (
            pos[0] + box_dim[0] <= placed_pos[0]
            or pos[0] >= placed_pos[0] + placed_dim[0]
            or pos[1] + box_dim[1] <= placed_pos[1]
            or pos[1] >= placed_pos[1] + placed_dim[1]
            or pos[2] + box_dim[2] <= placed_pos[2]
            or pos[2] >= placed_pos[2] + placed_dim[2]
        ):
            return True
    return False


def gen_cand_pos(
    box_dim: np.ndarray, placed: List[PlacedBox], bin_dim: np.ndarray
) -> List[np.ndarray]:
    """Generate candidate positions for placing the box."""
    candidates = [np.array([0.0, 0.0, 0.0])]

    for pos, dim in placed:
        for i in range(3):
            new_pos = pos.copy()
            new_pos[i] += dim[i]
            
            if is_valid_pos(new_pos, box_dim, bin_dim) and not check_overlap(
                new_pos, box_dim, placed
            ):
                candidates.append(new_pos)
    return candidates


def greedy_pos(
    box_dim: np.ndarray, bin_dim: np.ndarray, placed: List[PlacedBox]
) -> Optional[np.ndarray]:
    """Find a valid position using a simple greedy strategy."""
    candidates = gen_cand_pos(box_dim, placed, bin_dim)
    return candidates[0] if candidates else None


def crossover(
    parent1: Chromosome, parent2: Chromosome, boxes: List, bin_inst
) -> Chromosome:
    """Crossover two chromosomes to produce a valid child."""
    bin_dim = bin_inst.dim
    placed: List[PlacedBox] = []
    child: Chromosome = []

    for i, box in enumerate(boxes):
        box_dim = box.dim
        g1, g2 = parent1[i], parent2[i]
        s1, pos1 = g1[0], np.array(g1[1:])
        s2, pos2 = g2[0], np.array(g2[1:])
        candidate_pos: Optional[np.ndarray] = None

        if (
            s1 and is_valid_pos(pos1, box_dim, bin_dim)
            and not check_overlap(pos1, box_dim, placed)
        ):
            candidate_pos = pos1
        elif (
            s2 and is_valid_pos(pos2, box_dim, bin_dim)
            and not check_overlap(pos2, box_dim, placed)
        ):
            candidate_pos = pos2
        else:
            candidate_pos = greedy_pos(box_dim, bin_dim, placed)

        if candidate_pos is not None:
            placed.append((candidate_pos, box_dim))
            child.append([1.0, *candidate_pos])
        else:
            child.append(UNPLACED)

    return child


def repair(chromosome: Chromosome, boxes: List, bin_inst) -> Chromosome:
    """Repair an invalid chromosome by adjusting or removing boxes."""
    bin_dim = bin_inst.dim
    repaired: Chromosome = []
    placed: List[PlacedBox] = []

    for gene, box in zip(chromosome, boxes):
        selected, x, y, z = gene
        box_dim = box.dim

        if not selected:
            repaired.append(UNPLACED)
            continue

        pos = np.array([x, y, z])
        if is_valid_pos(pos, box_dim, bin_dim) and not check_overlap(pos, box_dim, placed):
            repaired.append([1.0, *pos])
            placed.append((pos, box_dim))
        else:
            new_pos = greedy_pos(box_dim, bin_dim, placed)
            if new_pos is not None:
                repaired.append([1.0, *new_pos])
                placed.append((new_pos, box_dim))
            else:
                repaired.append(UNPLACED)

    return repaired