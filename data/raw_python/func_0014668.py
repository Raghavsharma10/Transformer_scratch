def remove(self):
        '''
            remove - Will remove this node from its parent, if it has a parent (thus taking it out of the HTML tree)

                NOTE: If you are using an IndexedAdvancedHTMLParser, calling this will NOT update the index. You MUST call
                  reindex method manually.

            @return <bool> - While JS DOM defines no return for this function, this function will return True if a
               remove did happen, or False if no parent was set.
        '''
        if self.parentNode:
            self.parentNode.removeChild(self)
            # self.parentNode will now be None by 'removeChild' method
            return True
        return False