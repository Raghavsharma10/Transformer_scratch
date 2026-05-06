def from_dict(self, d: Dict[str, Any]) -> None:
        """Load values from a dict."""
        for key, value in d.items():
            if key.isupper():
                self._setattr(key, value)

        logger.info("Config is loaded from dict: %r", d)