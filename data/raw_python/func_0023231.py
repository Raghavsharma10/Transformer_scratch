def _remove_dep(self, dep):
        """ Decrement the reference count for *dep*. If the reference count 
        reaches 0, then the dependency is removed and its *changed* event is
        disconnected.
        """
        refcount = self._deps[dep]
        if refcount == 1:
            self._deps.pop(dep)
            dep._dependents.pop(self)
        else:
            self._deps[dep] -= 1