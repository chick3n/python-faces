"""
Sync Interface
"""

class SyncInterface:
    """
    SyncInterface
    """
    def sync(self) -> tuple:
        """
        sync
        """
        raise NotImplementedError()

    def move(self, src: str, dst: str) -> str:
        """
        save
        """
        raise NotImplementedError()
