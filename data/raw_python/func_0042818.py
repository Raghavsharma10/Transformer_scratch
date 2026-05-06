def get_index(self, index, type, alias=None, typed=None, read_only=True, kwargs=None):
        """
        TESTS THAT THE INDEX EXISTS BEFORE RETURNING A HANDLE
        """
        if kwargs.tjson != None:
            Log.error("used `typed` parameter, not `tjson`")
        if read_only:
            # GET EXACT MATCH, OR ALIAS
            aliases = wrap(self.get_aliases())
            if index in aliases.index:
                pass
            elif index in aliases.alias:
                match = [a for a in aliases if a.alias == index][0]
                kwargs.alias = match.alias
                kwargs.index = match.index
            else:
                Log.error("Can not find index {{index_name}}", index_name=kwargs.index)
            return Index(kwargs=kwargs, cluster=self)
        else:
            # GET BEST MATCH, INCLUDING PROTOTYPE
            best = self.get_best_matching_index(index, alias)
            if not best:
                Log.error("Can not find index {{index_name}}", index_name=kwargs.index)

            if best.alias != None:
                kwargs.alias = best.alias
                kwargs.index = best.index
            elif kwargs.alias == None:
                kwargs.alias = kwargs.index
                kwargs.index = best.index

            return Index(kwargs=kwargs, cluster=self)