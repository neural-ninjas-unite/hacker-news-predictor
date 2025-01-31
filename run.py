import os
import sys

def setup_python_path():
    """Set up the Python path to include the src directory."""
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Add the src directory to the Python path
    src_path = os.path.join(current_dir, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)
        print(f"Added {src_path} to Python path")

if __name__ == "__main__":
    # If running as a script
    setup_python_path()
    from main import main
    main() 