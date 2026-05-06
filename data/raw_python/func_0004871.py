def ReadList(self, *branches, **kwargs):
        """
Same as `phi.dsl.Expression.List` but any string argument `x` is translated to `Read(x)`.
        """
        branches = map(lambda x: E.Read(x) if isinstance(x, str) else x, branches)

        return self.List(*branches, **kwargs)