def _handleClassAttr(self):
        '''
            _handleClassAttr - Hack to ensure "class" and "style" show up in attributes when classes are set,
                and doesn't when no classes are present on associated tag.

                TODO: I don't like this hack.
        '''
        if len(self.tag._classNames) > 0:
            dict.__setitem__(self, "class", self.tag.className)
        else:
            try:
                dict.__delitem__(self, "class")
            except:
                pass

        styleAttr = self.tag.style
        if styleAttr.isEmpty() is False:
            dict.__setitem__(self, "style", styleAttr)
        else:
            try:
                dict.__delitem__(self, "style")
            except:
                pass