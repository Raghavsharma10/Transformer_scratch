def getAllChildNodes(self):
        '''
            getAllChildNodes - Gets all the children, and their children, 
               and their children, and so on, all the way to the end as a TagCollection.
               
               Use .childNodes for a regular list

            @return TagCollection<AdvancedTag> - A TagCollection of all children (and their children recursive)
        '''

        ret = TagCollection()

        # Scan all the children of this node
        for child in self.children:
            # Append each child
            ret.append(child)

            # Append children's children recursive
            ret += child.getAllChildNodes()

        return ret