def getPeers(self):
        '''
            getPeers - Get elements who share a parent with this element

            @return - TagCollection of elements
        '''
        parentNode = self.parentNode
        # If no parent, no peers
        if not parentNode:
            return None

        peers = parentNode.children

        # Otherwise, get all children of parent excluding this node
        return TagCollection([peer for peer in peers if peer is not self])