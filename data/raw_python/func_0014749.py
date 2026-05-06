def _setTag(self, tag):
        '''
            _setTag - INTERNAL METHOD. Associated a given AdvancedTag to this attributes dict.

                        If bool(#tag) is True, will set the weakref to that tag.

                        Otherwise, will clear the reference

                      @param tag <AdvancedTag/None> - Either the AdvancedTag to associate, or None to clear current association
        '''
        if tag:
            self._tagRef = weakref.ref(tag)
        else:
            self._tagRef = None