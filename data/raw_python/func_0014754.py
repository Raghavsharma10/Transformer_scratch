def _ensureHtmlAttribute(self):
        '''
            _ensureHtmlAttribute - INTERNAL METHOD. 
                                    Ensure the "style" attribute is present in the html attributes when
                                        is has a value, and absent when it does not.

              This requires special linkage.
        '''
        tag = self.tag

        if tag:
            styleDict = self._styleDict
            tagAttributes = tag._attributes

            # If this is called before we have _attributes setup
            if not issubclass(tagAttributes.__class__, SpecialAttributesDict):
                return

            # If we have any styles set, ensure we have the style="whatever" in the HTML representation,
            #   otherwise ensure we don't have style="" 
            if not styleDict:
                tagAttributes._direct_del('style')
            else: #if 'style' not in tagAttributes.keys():
                tagAttributes._direct_set('style', self)