def compute(self, *args, **kwargs)->[Any, None]:
        """Compose and evaluate the function.
        """
        return super().compute(
            self.compose, *args, **kwargs
        )