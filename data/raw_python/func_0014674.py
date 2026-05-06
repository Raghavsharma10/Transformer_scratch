def removeChild(self, child):
        '''
            removeChild - Remove a child tag, if present.

                @param child <AdvancedTag> - The child to remove

                @return - The child [with parentNode cleared] if removed, otherwise None.

                NOTE: This removes a tag. If removing a text block, use #removeText function.
                  If you need to remove an arbitrary block (text or AdvancedTag), @see removeBlock

                Removing multiple children? @see removeChildren
        '''
        try:
            # Remove from children and blocks
            self.children.remove(child)
            self.blocks.remove(child)

            # Clear parent node association on child
            child.parentNode = None

            # Clear document reference on removed child and all children thereof
            child.ownerDocument = None
            for subChild in child.getAllChildNodes():
                subChild.ownerDocument = None
            return child
        except ValueError:
            # TODO: What circumstances cause this to be raised? Is it okay to have a partial remove?
            #
            #  Is it only when "child" is not found? Should that just be explicitly tested?
            return None