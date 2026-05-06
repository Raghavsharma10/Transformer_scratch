def setTag(self, tag):
        '''
            setTag - Set the tag association for this style.
            
              This will handle the underlying weakref to the tag.

              Call setTag(None) to clear the association, otherwise setTag(tag) to associate this style to that tag.


                @param tag <AdvancedTag/None> - The new association. If None, the association is cleared, otherwise the passed tag
                    becomes associated with this style.

        '''
                
        if tag:
            self._tagRef = weakref.ref(tag)
        else:
            self._tagRef = None