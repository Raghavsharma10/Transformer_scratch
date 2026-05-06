def two_way_difference(self, b, extra_add = (), extra_remove = ()):
        """
        Return (self - b, b - self)
        """
        if self is b:
            return ((), ())
        if isinstance(b, DiffRef_):
            extra_remove = extra_remove + b.add
            b = b.origin
        if extra_add == extra_remove:
            extra_add = extra_remove = ()
        if isinstance(b, Diff_):
            if self.base is b.base:
                first = self.add + b.remove
                second = self.remove + b.add
            elif self.base is b:
                first = self.add
                second = self.remove
            elif b.base is self:
                first = b.remove
                second = b.add
            else:
                first = self
                second = b
        else:
            first = self
            second = b
        if not first and not extra_add:
            return ((), tuple(second) + tuple(extra_remove))
        elif not second and not extra_remove:
            return (tuple(first) + tuple(extra_add), ())
        else:
            first = set(first)
            first.update(extra_add)
            second = set(second)
            second.update(extra_remove)
            return tuple(first.difference(second)), tuple(second.difference(first))