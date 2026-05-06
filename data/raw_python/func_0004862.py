def Then0(self, f, *args, **kwargs):
        """
`Then0(f, ...)` is equivalent to `ThenAt(0, f, ...)`. Checkout `phi.builder.Builder.ThenAt` for more information.
        """
        return self.ThenAt(0, f, *args, **kwargs)