def getAllChildNodeUids(self):
        '''
            getAllChildNodeUids - Returns all the unique internal IDs for all children, and there children, 
              so on and so forth until the end.

              For performing "contains node" kind of logic, this is more efficent than copying the entire nodeset

            @return set<uuid.UUID> A set of uuid objects
        '''
        ret = set()

        # Iterate through all children
        for child in self.children:
            # Add child's uid
            ret.add(child.uid)
            # Add child's children's uid and their children, recursive
            ret.update(child.getAllChildNodeUids())

        return ret