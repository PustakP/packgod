import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from genetic_algorithm import GeneticAlgorithm, run_ga
from generate_instances import Bin, Box
from crossover import Chromosome

def create_box_mesh(pos, dim, color='blue', alpha=0.7):
    """create 3d box mesh for plotly"""
    x, y, z = pos
    dx, dy, dz = dim
    
    # define 8 vertices of box
    vertices = np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],  # bottom
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]  # top
    ])
    
    # define faces (triangles)
    faces = [
        [0,1,2], [0,2,3],  # bottom
        [4,7,6], [4,6,5],  # top  
        [0,4,5], [0,5,1],  # front
        [2,6,7], [2,7,3],  # back
        [0,3,7], [0,7,4],  # left
        [1,5,6], [1,6,2]   # right
    ]
    
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
        color=color, opacity=alpha, showscale=False
    )

def visualize_solution(bin_inst: Bin, boxes: List[Box], chromosome: Chromosome, title="3d bin packing"):
    """visualize single solution"""
    fig = go.Figure()
    
    # add bin outline
    bin_x, bin_y, bin_z = bin_inst.dim
    bin_outline = go.Scatter3d(
        x=[0, bin_x, bin_x, 0, 0, 0, bin_x, bin_x, 0, 0, bin_x, bin_x, bin_x, bin_x, 0, 0],
        y=[0, 0, bin_y, bin_y, 0, 0, 0, bin_y, bin_y, 0, 0, bin_y, bin_y, 0, 0, bin_y],
        z=[0, 0, 0, 0, 0, bin_z, bin_z, bin_z, bin_z, bin_z, bin_z, bin_z, 0, 0, 0, 0],
        mode='lines',
        line=dict(color='black', width=3),
        name='bin'
    )
    fig.add_trace(bin_outline)
    
    # add placed boxes
    colors = px.colors.qualitative.Set3
    placed_count = 0
    
    for i, (gene, box) in enumerate(zip(chromosome, boxes)):
        if gene[0] > 0:  # placed
            pos = np.array(gene[1:4])
            color = colors[i % len(colors)]
            
            box_mesh = create_box_mesh(pos, box.dim, color=color, alpha=0.7)
            box_mesh.name = f'box-{box.id}'
            fig.add_trace(box_mesh)
            placed_count += 1
    
    fig.update_layout(
        title=f"{title} - {placed_count}/{len(boxes)} boxes placed",
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=False
    )
    
    return fig

def plot_fitness_evolution(ga: GeneticAlgorithm):
    """plot fitness over generations"""
    gens = [h['generation'] for h in ga.fitness_history]
    best = [h['best_fitness'] for h in ga.fitness_history]
    avg = [h['avg_fitness'] for h in ga.fitness_history]
    worst = [h['worst_fitness'] for h in ga.fitness_history]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best, 'g-', label='best', linewidth=2)
    ax.plot(gens, avg, 'b-', label='avg', alpha=0.7)
    ax.plot(gens, worst, 'r-', label='worst', alpha=0.5)
    
    ax.set_xlabel('generation')
    ax.set_ylabel('fitness')
    ax.set_title('fitness evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_generation_comparison(ga: GeneticAlgorithm, generations_to_show=[0, 25, 50, 75, 99]):
    """create subplot comparing solutions across generations"""
    fig = make_subplots(
        rows=1, cols=len(generations_to_show),
        specs=[[{'type': 'scatter3d'} for _ in generations_to_show]],
        subplot_titles=[f'gen {g}' for g in generations_to_show]
    )
    
    colors = px.colors.qualitative.Set3
    
    for col, gen in enumerate(generations_to_show, 1):
        if gen < len(ga.best_solutions):
            chromosome = ga.best_solutions[gen]
            
            # bin outline
            bin_x, bin_y, bin_z = ga.bin_inst.dim
            fig.add_trace(
                go.Scatter3d(
                    x=[0, bin_x, bin_x, 0, 0, 0, bin_x, bin_x, 0, 0, bin_x, bin_x, bin_x, bin_x, 0, 0],
                    y=[0, 0, bin_y, bin_y, 0, 0, 0, bin_y, bin_y, 0, 0, bin_y, bin_y, 0, 0, bin_y],
                    z=[0, 0, 0, 0, 0, bin_z, bin_z, bin_z, bin_z, bin_z, bin_z, bin_z, 0, 0, 0, 0],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                ),
                row=1, col=col
            )
            
            # boxes
            for i, (gene, box) in enumerate(zip(chromosome, ga.boxes)):
                if gene[0] > 0:  # placed
                    pos = np.array(gene[1:4])
                    color = colors[i % len(colors)]
                    
                    box_mesh = create_box_mesh(pos, box.dim, color=color, alpha=0.6)
                    box_mesh.showlegend = False
                    fig.add_trace(box_mesh, row=1, col=col)
    
    fig.update_layout(
        title="evolution across generations",
        height=400,
        showlegend=False
    )
    
    return fig

def analyze_packing_efficiency(ga: GeneticAlgorithm):
    """analyze and visualize packing stats"""
    final_solution = ga.best_solutions[-1]
    
    # calc stats
    total_box_vol = sum(box.volume() for box in ga.boxes)
    placed_vol = sum(box.volume() for box, gene in zip(ga.boxes, final_solution) if gene[0] > 0)
    bin_vol = ga.bin_inst.volume()
    
    efficiency_metrics = {
        'boxes_placed': sum(1 for gene in final_solution if gene[0] > 0),
        'total_boxes': len(ga.boxes),
        'placement_rate': sum(1 for gene in final_solution if gene[0] > 0) / len(ga.boxes),
        'volume_utilization': placed_vol / bin_vol,
        'packing_efficiency': placed_vol / total_box_vol
    }
    
    # create metrics plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # boxes placed
    ax1.bar(['placed', 'unplaced'], 
            [efficiency_metrics['boxes_placed'], 
             efficiency_metrics['total_boxes'] - efficiency_metrics['boxes_placed']],
            color=['green', 'red'], alpha=0.7)
    ax1.set_title('box placement')
    ax1.set_ylabel('count')
    
    # volume comparison
    ax2.bar(['used', 'unused bin', 'unplaced boxes'], 
            [placed_vol, bin_vol - placed_vol, total_box_vol - placed_vol],
            color=['blue', 'gray', 'orange'], alpha=0.7)
    ax2.set_title('volume distribution')
    ax2.set_ylabel('volume')
    
    # efficiency ratios
    ratios = ['placement\nrate', 'volume\nutilization', 'packing\nefficiency']
    values = [efficiency_metrics['placement_rate'], 
              efficiency_metrics['volume_utilization'],
              efficiency_metrics['packing_efficiency']]
    ax3.bar(ratios, values, color=['purple', 'teal', 'brown'], alpha=0.7)
    ax3.set_title('efficiency metrics')
    ax3.set_ylabel('ratio')
    ax3.set_ylim(0, 1)
    
    # fitness over time
    gens = [h['generation'] for h in ga.fitness_history]
    fitness = [h['best_fitness'] for h in ga.fitness_history]
    ax4.plot(gens, fitness, 'g-', linewidth=2)
    ax4.set_title('fitness evolution')
    ax4.set_xlabel('generation')
    ax4.set_ylabel('fitness')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, efficiency_metrics

def run_full_visualization():
    """run complete ga and create all visualizations"""
    print("running genetic algorithm...")
    ga, best_solution = run_ga()
    
    print("\ncreating visualizations...")
    
    # 3d solution viz
    solution_fig = visualize_solution(ga.bin_inst, ga.boxes, best_solution, "best solution")
    solution_fig.show()
    
    # fitness evolution
    fitness_fig = plot_fitness_evolution(ga)
    fitness_fig.show()
    
    # generation comparison
    comparison_fig = create_generation_comparison(ga)
    comparison_fig.show()
    
    # efficiency analysis
    efficiency_fig, metrics = analyze_packing_efficiency(ga)
    efficiency_fig.show()
    
    print("\nfinal metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return ga, best_solution

if __name__ == "__main__":
    run_full_visualization()
