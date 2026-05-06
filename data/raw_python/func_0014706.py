def setStyles(self, styleUpdatesDict):
        '''
            setStyles - Sets one or more style params. 
                This all happens in one shot, so it is much much faster than calling setStyle for every value.

                To remove a style, set its value to empty string.
                When all styles are removed, the "style" attribute will be nullified.

            @param styleUpdatesDict - Dictionary of attribute : value styles.

            @return - String of current value of "style" after change is made.
        '''
        setStyleMethod = self.setStyle
        for newName, newValue in styleUpdatesDict.items():
            setStyleMethod(newName, newValue)

        return self.style