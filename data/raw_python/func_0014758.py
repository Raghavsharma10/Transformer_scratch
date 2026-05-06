def styleToDict(styleStr):
        '''
            getStyleDict - Gets a dictionary of style attribute/value pairs.

              NOTE: dash-names (like padding-top) are used here

            @return - OrderedDict of "style" attribute.
        '''
        styleStr = styleStr.strip()
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