def setStyle(self, styleName, styleValue):
        '''
            setStyle - Sets a style param. Example: "display", "block"

                If you need to set many styles on an element, use setStyles instead. 
                It takes a dictionary of attribute, value pairs and applies it all in one go (faster)

                To remove a style, set its value to empty string.
                When all styles are removed, the "style" attribute will be nullified.

            @param styleName - The name of the style element
            @param styleValue - The value of which to assign the style element

            @return - String of current value of "style" after change is made.
        '''
        myAttributes = self._attributes

        if 'style' not in myAttributes:
            myAttributes['style'] = "%s: %s" %(styleName, styleValue)
        else:
            setattr(myAttributes['style'], styleName, styleValue)