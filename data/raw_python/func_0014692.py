def getAllNodeUids(self):
        '''
            getAllNodeUids - Returns all the unique internal IDs from getAllChildNodeUids, but also includes this tag's uid

            @return set<uuid.UUID> A set of uuid objects
        '''
        # Start with a set including this tag's uuid
        ret = { self.uid }

        ret.update(self.getAllChildNodeUids())

        return ret