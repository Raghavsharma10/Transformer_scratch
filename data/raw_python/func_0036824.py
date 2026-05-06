def from_json(cls, string):
        """Create AnimeFiles from JSON string."""
        obj = json.loads(string)
        return cls(obj['regexp'], obj['files'])