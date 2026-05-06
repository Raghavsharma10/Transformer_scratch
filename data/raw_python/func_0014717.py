def getPeersByName(self, name):
        '''
            getPeersByName - Gets peers (elements on same level) with a given name

            @param name - Name to match

            @return - None if no parent element (error condition), otherwise a TagCollection of peers that matched.
        '''
        peers = self.peers
        if peers is None:
            return None
        return TagCollection([peer for peer in peers if peer.name == name])