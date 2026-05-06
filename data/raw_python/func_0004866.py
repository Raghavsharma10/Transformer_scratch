def Then4(self, f, arg1, arg2, arg3, *args, **kwargs):
        """
`Then4(f, ...)` is equivalent to `ThenAt(4, f, ...)`. Checkout `phi.builder.Builder.ThenAt` for more information.
        """
        args = (arg1, arg2, arg3) + args
        return self.ThenAt(4, f, *args, **kwargs)