def from_dict(cls, obj_dict: Dict[str, Any]) -> "IterationRecord":
        """Get object back from dict."""
        obj = cls()
        for key, item in obj_dict.items():
            obj.__dict__[key] = item

        return obj