def get(self, path):
        """ Get a transform from the cache that maps along *path*, which must
        be a list of Transforms to apply in reverse order (last transform is
        applied first).

        Accessed items have their age reset to 0.
        """
        key = tuple(map(id, path))
        item = self._cache.get(key, None)
        if item is None:
            logger.debug("Transform cache miss: %s", key)
            item = [0, self._create(path)]
            self._cache[key] = item
        item[0] = 0  # reset age for this item

        # make sure the chain is up to date
        #tr = item[1]
        #for i, node in enumerate(path[1:]):
        #    if tr.transforms[i] is not node.transform:
        #        tr[i] = node.transform

        return item[1]