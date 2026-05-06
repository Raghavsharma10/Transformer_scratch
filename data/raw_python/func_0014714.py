def getPeersCustomFilter(self, filterFunc):
        '''
            getPeersCustomFilter - Get elements who share a parent with this element and also pass a custom filter check

                @param filterFunc <lambda/function> - Passed in an element, and returns True if it should be treated as a match, otherwise False.

                @return <TagCollection> - Resulting peers, or None if no parent node.
        '''
        peers = self.peers
        if peers is None:
            return None

        return TagCollection([peer for peer in peers if filterFunc(peer) is True])