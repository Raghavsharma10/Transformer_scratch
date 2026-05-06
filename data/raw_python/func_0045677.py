def from_file(cls, path, **kwargs):
        """Create an editor instance from a file on disk."""
        lines = lines_from_file(path)
        if 'meta' not in kwargs:
            kwargs['meta'] = {'from': 'file'}
        kwargs['meta']['filepath'] = path
        return cls(lines, **kwargs)