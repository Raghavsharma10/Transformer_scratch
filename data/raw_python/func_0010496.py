def from_env_vars(self) -> None:
        """Load values from environment variables.
        Keys must start with `KUYRUK_`."""
        for key, value in os.environ.items():
            if key.startswith('KUYRUK_'):
                key = key[7:]
                if hasattr(Config, key):
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass

                    self._setattr(key, value)