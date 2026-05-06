def containsUid(self, uid):
        '''
            Check if #uid is found anywhere within this element tree

            @param uid <uuid.UUID> - Uid

            @return <bool> - If #uid is found within this tree
        '''
        for rootNode in self.getRootNodes():
            if rootNode.containsUid(uid):
                return True

        return False