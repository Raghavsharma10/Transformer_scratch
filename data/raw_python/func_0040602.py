def intersection(self,other):
        "intersection with another patch"
        res = {}
        if set(self.sets.keys()) != set(other.sets.keys()):
            raise KeyError('Incompatible patches in intersection')
        for name,s1 in self.sets.items():
            s2 = other.sets[name]
            res[name] = s1.intersection(s2)
        return Patch(res)