def Then2(self, f, arg1, *args, **kwargs):
        """
`Then2(f, ...)` is equivalent to `ThenAt(2, f, ...)`. Checkout `phi.builder.Builder.ThenAt` for more information.
        """
        args = (arg1,) + args
        return self.ThenAt(2, f, *args, **kwargs)