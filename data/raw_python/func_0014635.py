def getElementsWithAttrValues(self, attrName, attrValues, root='root'):
        '''
            getElementsWithAttrValues - Returns elements with an attribute, named by #attrName contains one of the values in the list, #values

            @param attrName <lowercase str> - A lowercase attribute name
            @param attrValues set<str> - A set of all valid values. 


            @return - TagCollection of all matching elements

        '''
        (root, isFromRoot) = self._handleRootArg(root)

        if type(attrValues) != set:
            attrValues = set(attrValues)
        
        return root.getElementsWithAttrValues(attrName, attrValues)