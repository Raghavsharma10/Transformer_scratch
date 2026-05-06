def Then3(self, f, arg1, arg2, *args, **kwargs):
        """
`Then3(f, ...)` is equivalent to `ThenAt(3, f, ...)`. Checkout `phi.builder.Builder.ThenAt` for more information.
        """
        args = (arg1, arg2) + args
        return self.ThenAt(3, f, *args, **kwargs)