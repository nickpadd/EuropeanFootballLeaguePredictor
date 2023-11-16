import os
from loguru import logger 

class PathHandler:
    """Handles directories
    Args:
        paths: A string indicating a path
    Returns:
        None
    """

    def __init__(self, path: str):
        self.path = path

    def create_paths_if_not_exists(self) -> None:
        """Creates directories if not exist"""
        if not os.path.exists(self.path):
            # Create a new directory because it does not exist
            os.makedirs(self.path)
            logger.debug(f'Created {self.path}')

    def remove_paths_if_exists(self) -> None:
        """Deletes directories if exist"""
        if os.path.exists(self.path):
            # Create a new directory because it does not exist
            os.rmdir(self.path)