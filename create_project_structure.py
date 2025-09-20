import os
from pathlib import Path


def create_project_structure():
    """
    Creates the folder and file structure for the project as specified.
    """
    base_dir = Path.cwd()

    # Define the structure to be created
    structure = {
        "app": {
            "backend": [
                "__init__.py",
                "agent.py",
                "document_parser.py",
                "main.py",
                "rag_pipeline.py",
            ],
            "frontend": [
                "__init__.py",
                "app.py",
            ],
            "Dockerfile.backend": None,
            "Dockerfile.frontend": None,
        },
        "data": {},  # Use an empty dictionary to represent an empty directory
        "docker-compose.yml": None,
        "requirements.txt": None,
        "README.md": None,
    }

    # Helper function to create directories and files recursively
    def create_items(current_path, items):
        if isinstance(items, dict):
            for name, content in items.items():
                new_path = current_path / name

                # Check if a file with the same name exists
                if os.path.isfile(new_path):
                    print(f"File already exists, skipping: {new_path}")
                    continue

                if isinstance(content, dict):
                    # It's a directory with sub-items
                    os.makedirs(new_path, exist_ok=True)
                    print(f"Created directory: {new_path}")
                    create_items(new_path, content)  # Recursive call
                elif isinstance(content, list):
                    # It's a directory with files
                    os.makedirs(new_path, exist_ok=True)
                    print(f"Created directory: {new_path}")
                    for file_name in content:
                        file_path = new_path / file_name
                        file_path.touch(exist_ok=True)
                        print(f"  Created file: {file_path}")
                else:
                    # It's a simple file
                    new_path.touch(exist_ok=True)
                    print(f"  Created file: {new_path}")

    create_items(base_dir, structure)


if __name__ == "__main__":
    create_project_structure()
