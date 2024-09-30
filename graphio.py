import networkx as nx
from pydantic import BaseModel, Field, field_validator, ValidationError
from tqdm import tqdm
import os

# Pydantic model for file validation
class FilePathModel(BaseModel):
    filepath: str = Field(...)

    @field_validator('filepath')
    def validate_filepath(cls, v: str) -> str:
        if not v.endswith('.gml'):
            raise ValueError('File path must end with .gml')
        return v.strip()

def split_file_path(filepath: str):
    """Splits the file path into directory and filename."""
    try:
        validated_filepath = FilePathModel(filepath=filepath).filepath
        directory = os.path.dirname(validated_filepath)
        filename = os.path.basename(validated_filepath)
        return directory, filename
    except ValidationError as e:
        print(f"Error: {e}")
        return None, None

# Function to save a NetworkX graph as a GML file
def save_gml(graph, filepath='graph.gml'):
    """Saves the NetworkX graph as a GML file."""
    directory, filename = split_file_path(filepath)

    if directory is None or filename is None:
        return  # Exit if there was an error in splitting the path

    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Error: The directory {directory} does not exist.")
        return

    # Attempt to save the graph
    try:
        with tqdm(total=1, desc="Saving graph") as pbar:
            full_path = os.path.join(directory, filename)
            nx.write_gml(graph, full_path)
            pbar.update(1)
        print(f"Graph has been saved as {full_path}")
    except Exception as e:
        print(f"Failed to save graph: {e}")

# Function to read a NetworkX graph from a GML file
def read_gml(filepath):
    """Reads a NetworkX graph from a GML file."""
    directory, filename = split_file_path(filepath)

    if directory is None or filename is None:
        return None  # Exit if there was an error in splitting the path

    # Validate that the file exists
    full_path = os.path.join(directory, filename)
    
    if not os.path.isfile(full_path):
        print(f"Error: The file {full_path} does not exist.")
        return None

    # Attempt to read the graph
    try:
        with tqdm(total=1, desc="Loading graph") as pbar:
            G = nx.read_gml(full_path)
            pbar.update(1)
        print(f"Graph {full_path} has been loaded")
        return G
    except Exception as e:
        print(f"Failed to load graph: {e}")
        return None

# Example usage (uncomment to test):
# g = nx.Graph()
# g.add_edges_from([(1, 2), (2, 3)])
# save_gml(g, '/home/riju279/Documents/Code/DSPyG/DSPy_scaper/my_graph.gml')
# loaded_graph = read_gml('/home/riju279/Documents/Code/DSPyG/DSPy_scaper/my_graph.gml')
