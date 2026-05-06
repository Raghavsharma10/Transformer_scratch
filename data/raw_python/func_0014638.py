def contains(self, em):
        '''
            Checks if #em is found anywhere within this element tree

            @param em <AdvancedTag> - Tag of interest

            @return <bool> - If element #em is within this tree
        '''
        for rootNode in self.getRootNodes():
            if rootNode.contains(em):
                return True

        return False