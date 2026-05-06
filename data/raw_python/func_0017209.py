def get_cached_content(self, store_attr=None, container_type=set, _store=None):
        """
        Returns a dictionary pointing to the preloaded content of each
        internal node under this tree. Such a dictionary is intended
        to work as a cache for operations that require many traversal
        operations.
        
        Parameters:
        -----------
        store_attr: 
            Specifies the node attribute that should be cached (i.e. name, 
            distance, etc.). When none, the whole node instance is cached.
        _store: (internal use)
        """
        if _store is None:
            _store = {}

        for ch in self.children:
            ch.get_cached_content(store_attr=store_attr,
                                  container_type=container_type,
                                  _store=_store)
        if self.children:
            val = container_type()
            for ch in self.children:
                if type(val) == list:
                    val.extend(_store[ch])
                if type(val) == set:
                    val.update(_store[ch])
            _store[self] = val
        else:
            if store_attr is None:
                val = self
            else:
                val = getattr(self, store_attr)
            _store[self] = container_type([val])
        return _store