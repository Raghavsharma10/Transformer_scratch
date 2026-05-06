def loadconfig(self, keysuffix, obj):
        """
        Copy all configurations from this node into obj
        """
        subtree = self.get(keysuffix)
        if subtree is not None and isinstance(subtree, ConfigTree):
            for k,v in subtree.items():
                if isinstance(v, ConfigTree):
                    if hasattr(obj, k) and not isinstance(getattr(obj, k), ConfigTree):
                        v.loadconfig(getattr(obj,k))
                    else:
                        setattr(obj, k, v)
                elif not hasattr(obj, k):
                    setattr(obj, k, v)