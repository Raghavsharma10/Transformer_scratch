def getPeersByAttr(self, attrName, attrValue):
        '''
            getPeersByAttr - Gets peers (elements on same level) which match an attribute/value combination.

            @param attrName - Name of attribute
            @param attrValue - Value that must match

            @return - None if no parent element (error condition), otherwise a TagCollection of peers that matched.
        '''
        peers = self.peers
        if peers is None:
            return None
        return TagCollection([peer for peer in peers if peer.getAttribute(attrName) == attrValue])