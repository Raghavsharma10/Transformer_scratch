def from_stream(cls, f, **kwargs):
        """Create an editor instance from a file stream."""
        lines = lines_from_stream(f)
        if 'meta' not in kwargs:
            kwargs['meta'] = {'from': 'stream'}
        kwargs['meta']['filepath'] = f.name if hasattr(f, 'name') else None
        return cls(lines, **kwargs)