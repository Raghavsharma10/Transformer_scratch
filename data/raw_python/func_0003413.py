def subtree(self, event, create = False):
        '''
        Find a subtree from an event
        '''
        current = self
        for i in range(self.depth, len(event.indices)):
            if not hasattr(current, 'index'):
                return current
            ind = event.indices[i]
            if create:
                current = current.index.setdefault(ind, EventTree(current, self.branch))
                current.parentIndex = ind
            else:
                current = current.index.get(ind)
                if current is None:
                    return None
        return current