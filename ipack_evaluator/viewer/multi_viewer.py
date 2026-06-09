from .viewer_base import ViewerBase


class MultiViewer():
    def __init__(self, viewers: list[ViewerBase]) -> None:
        self.viewers = viewers

    def __getattr__(self, name):
        """Dynamically delegate any method call to all viewers."""
        def delegated_method(*args, **kwargs):
            results = []
            for viewer in self.viewers:
                if hasattr(viewer, name):
                    result = getattr(viewer, name)(*args, **kwargs)
                    results.append(result)
            # Return the first non-None result, or None if all are None
            return next((r for r in results if r is not None), None)
        
        return delegated_method
