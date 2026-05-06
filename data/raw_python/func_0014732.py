def contains(self, em):
        '''
            contains - Check if #em occurs within any of the elements within this list, as themselves or as a child, any
               number of levels down.

               To check if JUST an element is contained within this list directly, use the "in" operator.
            
            @param em <AdvancedTag> - Element of interest

            @return <bool> - True if contained, otherwise False
        '''

        for node in self:
            if node.contains(em):
                return True

        return False