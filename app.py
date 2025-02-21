import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

def create_tree_layout_pollination(n_clones=60, spacing=6, rows=8, cols=8, pattern="grid"):
    """Create a tree-like layout for pollination analysis
    
    Parameters:
    - n_clones: number of clones to place
    - spacing: distance between adjacent trees in meters
    - rows, cols: dimensions of the grid
    - pattern: "grid" for perfect grid or "dense" for a more compact layout
    """
    # Initialize positions list
    positions = []
    
    # Create list of clone numbers
    clone_numbers = list(range(1, n_clones + 1))
    random.shuffle(clone_numbers)  # Randomize clone placement
    
    if pattern == "grid":
        # Distribute clones in a grid pattern with spacing
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < n_clones:
                    positions.append((j * spacing, (rows-1-i) * spacing, clone_numbers[idx]))
                    idx += 1
    
    elif pattern == "dense":
        # Create a more densely packed layout similar to the first image
        # Calculate grid dimensions needed to fit all clones (with 20% buffer)
        grid_size = int(np.ceil(np.sqrt(n_clones * 1.2)))
        cell_size = spacing / 1.5  # Make cells smaller to pack trees more densely
        
        # Generate positions with some randomness
        idx = 0
        while idx < n_clones:
            for i in range(grid_size):
                for j in range(grid_size):
                    # Add some randomness to position within cell
                    x = j * cell_size + random.uniform(-cell_size/4, cell_size/4)
                    y = (grid_size-1-i) * cell_size + random.uniform(-cell_size/4, cell_size/4)
                    
                    if idx < n_clones:
                        positions.append((x, y, clone_numbers[idx]))
                        idx += 1
    
    return positions

def analyze_pollination_tree_layout(positions, pollination_distance=18, analysis_type="distance"):
    """Analyze pollination capabilities
    
    Parameters:
    - positions: list of (x, y, clone_number) tuples
    - pollination_distance: maximum distance for pollination in meters
    - analysis_type: "distance" for distance-based or "cell" for cell-based analysis
    """
    # Dictionary to store neighboring clones for each clone
    pollination_pairs = defaultdict(set)
    
    if analysis_type == "distance":
        # Check distances between all pairs of trees
        for i, (x1, y1, clone1) in enumerate(positions):
            for j, (x2, y2, clone2) in enumerate(positions):
                if i != j:  # Don't compare a tree to itself
                    # Calculate Euclidean distance
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    # If within pollination distance, add to pollination pairs
                    if distance <= pollination_distance:
                        pollination_pairs[clone1].add(clone2)
    
    elif analysis_type == "cell":
        # First, determine the grid cell for each tree
        cell_size = pollination_distance / 3  # Approx 3 cells radius
        tree_cells = {}
        
        for i, (x, y, clone) in enumerate(positions):
            # Convert position to grid cell
            cell_x = int(x / cell_size)
            cell_y = int(y / cell_size)
            tree_cells[clone] = (cell_x, cell_y)
        
        # Now check if trees are within 3 cell radius
        for clone1, (cell_x1, cell_y1) in tree_cells.items():
            for clone2, (cell_x2, cell_y2) in tree_cells.items():
                if clone1 != clone2:
                    # Calculate cell distance
                    cell_distance = max(abs(cell_x2 - cell_x1), abs(cell_y2 - cell_y1))
                    if cell_distance <= 3:  # Within 3-cell radius
                        pollination_pairs[clone1].add(clone2)
    
    return pollination_pairs

def plot_tree_pollination_layout(positions, pollination_analysis=None, spacing=6, layout_type="grid", show_grid=True):
    """Plot the tree layout with pollination analysis
    
    Parameters:
    - positions: list of (x, y, clone_number) tuples
    - pollination_analysis: dictionary of pollination pairs
    - spacing: grid spacing
    - layout_type: "grid" for evenly spaced, "dense" for compact layout
    - show_grid: whether to show grid lines
    """
    # Extract coordinates and clone numbers
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    clone_numbers = [p[2] for p in positions]
    
    # Calculate grid dimensions
    max_x = max(x) + spacing/2
    max_y = max(y) + spacing/2
    
    if layout_type == "grid":
        rows = int(max_y / spacing) + 1
        cols = int(max_x / spacing) + 1
    else:
        # For dense layout, adjust grid lines
        cell_size = spacing / 1.5
        rows = int(max_y / cell_size) + 1
        cols = int(max_x / cell_size) + 1
    
    # Create figure
    plt.figure(figsize=(15, 15))
    
    # Add grid lines if requested
    if show_grid:
        if layout_type == "grid":
            for i in range(rows + 1):
                plt.axhline(y=i * spacing, color='gray', linestyle='--', alpha=0.3)
            for j in range(cols + 1):
                plt.axvline(x=j * spacing, color='gray', linestyle='--', alpha=0.3)
        else:
            # For dense layout, show a finer grid
            for i in range(rows + 1):
                plt.axhline(y=i * cell_size, color='gray', linestyle='--', alpha=0.2)
            for j in range(cols + 1):
                plt.axvline(x=j * cell_size, color='gray', linestyle='--', alpha=0.2)
    
    # Plot trees with green circles
    plt.scatter(x, y, s=400, c='lightgreen', marker='o', edgecolor='green')
    
    # Add clone numbers inside the circles
    for i, txt in enumerate(clone_numbers):
        plt.annotate(str(txt), (x[i], y[i]), 
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='center', fontsize=8,
                    color='black')
    
    # Set plot properties
    title_text = 'Clone Distribution with Pan-mixing Pollination\n'
    if layout_type == "grid":
        subtitle = 'Trees planted on regular grid, pollination range: 18m'
    else:
        subtitle = 'Trees planted in compact layout, pollination across 3-cell radius'
    
    plt.title(title_text + subtitle, pad=20, fontsize=14)
    plt.xlabel('Distance (meters)')
    plt.ylabel('Distance (meters)')
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Set limits
    plt.xlim(-spacing/2, max_x)
    plt.ylim(-spacing/2, max_y)
    
    # Add pollination analysis if provided
    if pollination_analysis:
        min_partners = min(len(partners) for partners in pollination_analysis.values())
        max_partners = max(len(partners) for partners in pollination_analysis.values())
        avg_partners = sum(len(partners) for partners in pollination_analysis.values()) / len(pollination_analysis)
        
        analysis_text = f'Pollination Analysis:\n' + \
                       f'Min potential partners per clone: {min_partners}\n' + \
                       f'Max potential partners per clone: {max_partners}\n' + \
                       f'Avg potential partners per clone: {avg_partners:.1f}'
        
        # Add box with analysis
        plt.text(max_x + 1, max_y/2, analysis_text,
                bbox=dict(facecolor='white', alpha=0.8),
                va='center', fontsize=10)
    
    # Save the plot
    if layout_type == "grid":
        plt.savefig('grid_pollination_layout.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('dense_pollination_layout.png', dpi=300, bbox_inches='tight')
    plt.close()

    return pollination_analysis

def visualize_pollination_connections(positions, pollination_pairs, clone_to_show=None, spacing=6):
    """Visualize pollination connections for a specific clone or all clones"""
    # Extract coordinates and clone numbers
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    clone_numbers = [p[2] for p in positions]
    
    # Calculate grid dimensions
    max_x = max(x) + spacing/2
    max_y = max(y) + spacing/2
    rows = int(max_y / spacing) + 1
    cols = int(max_x / spacing) + 1
    
    # Create figure
    plt.figure(figsize=(15, 15))
    
    # Add grid lines
    for i in range(rows + 1):
        plt.axhline(y=i * spacing, color='gray', linestyle='--', alpha=0.3)
    for j in range(cols + 1):
        plt.axvline(x=j * spacing, color='gray', linestyle='--', alpha=0.3)
    
    # Plot trees
    plt.scatter(x, y, s=400, c='lightgreen', marker='o', edgecolor='green')
    
    # Add clone numbers
    for i, txt in enumerate(clone_numbers):
        plt.annotate(str(txt), (x[i], y[i]), 
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='center', fontsize=8,
                    color='black')
    
    # If a specific clone is selected, highlight it and show connections
    if clone_to_show:
        # Find position of the selected clone
        clone_idx = clone_numbers.index(clone_to_show)
        clone_x, clone_y = x[clone_idx], y[clone_idx]
        
        # Highlight the selected clone
        plt.scatter([clone_x], [clone_y], s=500, c='red', marker='o', alpha=0.5)
        
        # Draw lines to all pollination partners
        for partner in pollination_pairs[clone_to_show]:
            partner_idx = clone_numbers.index(partner)
            partner_x, partner_y = x[partner_idx], y[partner_idx]
            
            plt.plot([clone_x, partner_x], [clone_y, partner_y], 'b-', alpha=0.2)
    
    # Set plot properties
    if clone_to_show:
        plt.title(f'Pollination Connections for Clone {clone_to_show}\n' +
                 f'Can pollinate with {len(pollination_pairs[clone_to_show])} other clones', 
                 pad=20, fontsize=14)
    else:
        plt.title('Clone Distribution with Pan-mixing Pollination\n' +
                 'Trees planted on 6m grid, pollination range: 18m', 
                 pad=20, fontsize=14)
    
    plt.xlabel('Distance (meters)')
    plt.ylabel('Distance (meters)')
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Set limits
    plt.xlim(-spacing/2, max_x)
    plt.ylim(-spacing/2, max_y)
    
    # Save the plot
    if clone_to_show:
        plt.savefig(f'pollination_connections_clone_{clone_to_show}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('pollination_layout.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to create and analyze layouts with different configurations
def generate_pollination_layout(layout_type="grid", analysis_type="distance", n_clones=60, 
                               pollination_distance=18, show_grid=True):
    """Generate a complete pollination layout with the specified parameters"""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate tree layout
    positions = create_tree_layout_pollination(n_clones=n_clones, pattern=layout_type)
    
    # Analyze pollination capabilities
    pollination_pairs = analyze_pollination_tree_layout(
        positions, 
        pollination_distance=pollination_distance,
        analysis_type=analysis_type
    )
    
    # Plot the layout with analysis
    plot_tree_pollination_layout(
        positions, 
        pollination_pairs, 
        layout_type=layout_type,
        show_grid=show_grid
    )
    
    # Verify pan-mixing capability
    all_clones = set(range(1, n_clones + 1))
    missing_connections = []
    
    for clone, partners in pollination_pairs.items():
        missing_partners = all_clones - partners - {clone}
        if missing_partners:
            missing_connections.append((clone, missing_partners))
            print(f"Clone {clone} cannot pollinate with clones: {missing_partners}")
        else:
            print(f"Clone {clone} can pollinate with all other clones")
    
    if not missing_connections:
        print("\nPerfect pan-mixing achieved: All clones can pollinate with all other clones")
    else:
        print(f"\n{len(missing_connections)} clones have incomplete pollination connections")
    
    if layout_type == "grid":
        print("\nGrid pollination layout saved as 'grid_pollination_layout.png'")
    else:
        print("\nDense pollination layout saved as 'dense_pollination_layout.png'")
    
    return positions, pollination_pairs

# Generate grid layout (like Image 2)
grid_positions, grid_pollination = generate_pollination_layout(
    layout_type="grid", 
    analysis_type="distance"
)

# Generate dense layout (like Image 1)
dense_positions, dense_pollination = generate_pollination_layout(
    layout_type="dense", 
    analysis_type="cell"
)

# Optionally show connections for a specific clone
# Uncomment to visualize connections for clone 1
# visualize_pollination_connections(grid_positions, grid_pollination, clone_to_show=1)