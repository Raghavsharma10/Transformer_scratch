def Then(self, f, *args, **kwargs):
        """
`Then(f, ...)` is equivalent to `ThenAt(1, f, ...)`. Checkout `phi.builder.Builder.ThenAt` for more information.
        """
        return self.ThenAt(1, f, *args, **kwargs)