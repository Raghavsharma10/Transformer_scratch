def setProperty(self, name, value):
        '''
            setProperty - Set a style property to a value.

                NOTE: To remove a style, use a value of empty string, or None

                 @param name <str> - The style name.

                    NOTE: The dash names are expected here, whereas dot-access expects the camel case names.

                      Example:  name="font-weight"  versus the dot-access  style.fontWeight

                 @param value <str> - The style value, or empty string to remove property
        '''
        styleDict = self._styleDict

        if value in ('', None):
            try:
                del styleDict[name]
            except KeyError:
                pass
        else:
            styleDict[name] = str(value)