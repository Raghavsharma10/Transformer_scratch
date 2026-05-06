def removeChildren(self, children):
        '''
            removeChildren - Remove multiple child AdvancedTags.

            @see removeChild

            @return list<AdvancedTag/None> - A list of all tags removed in same order as passed.
                Item is "None" if it was not attached to this node, and thus was not removed.
        '''
        ret = []

        for child in children:
            ret.append( self.removeChild(child) )

        return ret