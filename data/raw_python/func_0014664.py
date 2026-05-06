def cloneNode(self):
        '''
            cloneNode - Clone this node (tag name and attributes). Does not clone children.

            Tags will be equal according to isTagEqual method, but will contain a different internal
            unique id such tag origTag != origTag.cloneNode() , as is the case in JS DOM.
        '''
        return self.__class__(self.tagName, self.getAttributesList(), self.isSelfClosing)