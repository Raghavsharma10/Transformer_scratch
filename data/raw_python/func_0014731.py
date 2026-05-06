def getAllNodeUids(self):
        '''
            getAllNodeUids - Gets all the internal uids of all nodes, their children, and all their children so on..

              @return set<uuid.UUID>
        '''
        ret = set()

        for child in self:
            ret.update(child.getAllNodeUids())

        return ret