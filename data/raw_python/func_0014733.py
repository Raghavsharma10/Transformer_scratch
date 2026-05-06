def containsUid(self, uid):
        '''
            containsUid - Check if #uid is the uid (unique internal identifier) of any of the elements within this list,
              as themselves or as a child, any number of levels down.

           
            @param uid <uuid.UUID> - uuid of interest

            @return <bool> - True if contained, otherwise False
        '''
        for node in self:
            if node.containsUid(uid):
                return True

        return False