def subtree(self, matcher, create = False):
        '''
        Find a subtree from a matcher
        
        :param matcher: the matcher to locate the subtree. If None, return the root of the tree.
        
        :param create: if True, the subtree is created if not exists; otherwise return None if not exists
        '''
        if matcher is None:
            return self
        current = self
        for i in range(self.depth, len(matcher.indices)):
            ind = matcher.indices[i]
            if ind is None:
                # match Any
                if hasattr(current, 'any'):
                    current = current.any
                else:
                    if create:
                        cany = MatchTree(current)
                        cany.parentIndex = None
                        current.any = cany
                        current = cany
                    else:
                        return None
            else:
                current2 = current.index.get(ind)
                if current2 is None:
                    if create:
                        cind = MatchTree(current)
                        cind.parentIndex = ind 
                        current.index[ind] = cind
                        current = cind
                    else:
                        return None
                else:
                    current = current2
        return current