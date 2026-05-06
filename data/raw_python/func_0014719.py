def isTagEqual(self, other):
        '''
            isTagEqual - Compare if a tag contains the same tag name and attributes as another tag,

                i.e. if everything between < and > parts of this tag are the same.

                Does NOT compare children, etc. Does NOT compare if these are the same exact tag in the html (use regular == operator for that)

                So for example:

                    tag1 = document.getElementById('something')
                    tag2 = copy.copy(tag1)

                    tag1 == tag2          # This is False
                    tag1.isTagEqual(tag2) # This is True

                @return bool - True if tags have the same name and attributes, otherwise False
        '''
#        if type(other) != type(self):
#            return False

#       NOTE: Instead of type check,
#          just see if we can get the needed attributes in case subclassing
        try:
            if self.tagName != other.tagName:
                return False

            myAttributes = self._attributes
            otherAttributes = other._attributes

            attributeKeysSelf = list(myAttributes.keys())
            attributeKeysOther = list(otherAttributes.keys())
        except:
            return False

        # Check that we have all the same attribute names
        if set(attributeKeysSelf) != set(attributeKeysOther):
            return False

        for key in attributeKeysSelf:

            if myAttributes.get(key) != otherAttributes.get(key):
                return False

        return True