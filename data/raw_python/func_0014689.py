def containsUid(self, uid):
        '''
            containsUid - Check if the uid (unique internal ID) appears anywhere as a direct child to this node, or the node itself.

                @param uid <uuid.UUID> - uuid to check

            @return <bool> - True if #uid is this node's uid, or is the uid of any children at any level down
        '''
        # Check if this node is the match
        if self.uid == uid:
            return True

        # Scan all children
        for child in self.children:
            if child.containsUid(uid):
                return True

        return False