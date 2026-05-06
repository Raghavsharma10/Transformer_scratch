def dependencies(self, sort=False):
        """ Return all dependencies required to use this object. The last item 
        in the list is *self*.
        """
        alldeps = []
        if sort:
            def key(obj):
                # sort deps such that we get functions, variables, self.
                if not isinstance(obj, Variable):
                    return (0, 0)
                else:
                    return (1, obj.vtype)
            
            deps = sorted(self._deps, key=key)
        else:
            deps = self._deps
        
        for dep in deps:
            alldeps.extend(dep.dependencies(sort=sort))
        alldeps.append(self)
        return alldeps