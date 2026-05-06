def getAllNodes(self):
        '''
            getAllNodes - Get every element

            @return TagCollection<AdvancedTag>
        '''

        ret = TagCollection()

        for rootNode in self.getRootNodes():
            ret.append(rootNode)

            ret += rootNode.getAllChildNodes()

        return ret