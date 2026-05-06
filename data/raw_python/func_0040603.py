def iter_points(self):
        "returns a list of tuples of names and values"
        if not self.is_discrete():
            raise ValueError("Patch is not discrete")
        names = sorted(self.sets.keys())
        icoords = [self.sets[name].iter_members() for name in names]
        for coordinates in product(*icoords):
            yield tuple(zip(names,coordinates))