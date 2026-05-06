def getPeersByClassName(self, className):
        '''
            getPeersByClassName - Gets peers (elements on same level) with a given class name

            @param className - classname must contain this name

            @return - None if no parent element (error condition), otherwise a TagCollection of peers that matched.
        '''
        peers = self.peers
        if peers is None:
            return None
        return TagCollection([peer for peer in peers if peer.hasClass(className)])