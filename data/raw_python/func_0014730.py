def getAllNodes(self):
        '''
            getAllNodes - Gets all the nodes, and all their children for every node within this collection
        '''
        ret = TagCollection()

        for tag in self:
            ret.append(tag)
            ret += tag.getAllChildNodes()

        return ret