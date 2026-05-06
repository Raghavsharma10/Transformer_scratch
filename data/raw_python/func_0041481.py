def _path_hooks(cls, path):  # from importlib.PathFinder
        """Search sys.path_hooks for a finder for 'path'."""
        if sys.path_hooks is not None and not sys.path_hooks:
            warnings.warn('sys.path_hooks is empty', ImportWarning)
        for hook in sys.path_hooks:
            try:
                return hook(path)
            except ImportError:
                continue
        else:
            return None