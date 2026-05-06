def Then5(self, f, arg1, arg2, arg3, arg4, *args, **kwargs):
        """
`Then5(f, ...)` is equivalent to `ThenAt(5, f, ...)`. Checkout `phi.builder.Builder.ThenAt` for more information.
        """
        args = (arg1, arg2, arg3, arg4) + args
        return self.ThenAt(5, f, *args, **kwargs)