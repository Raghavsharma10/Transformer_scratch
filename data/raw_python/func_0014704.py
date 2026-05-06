def getStyleDict(self):
        '''
            getStyleDict - Gets a dictionary of style attribute/value pairs.

            @return - OrderedDict of "style" attribute.
        '''

        # TODO: This method is not used and does not appear in any tests.

        styleStr = (self.getAttribute('style') or '').strip()
        styles = styleStr.split(';') # Won't work for strings containing semicolon..
        styleDict = OrderedDict()
        for item in styles:
            try:
                splitIdx = item.index(':')
                name = item[:splitIdx].strip().lower()
                value = item[splitIdx+1:].strip()
                styleDict[name] = value
            except:
                continue

        return styleDict