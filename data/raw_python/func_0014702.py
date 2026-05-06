def addClass(self, className):
        '''
            addClass - append a class name to the end of the "class" attribute, if not present

                @param className <str> - The name of the class to add
        '''
        className = stripWordsOnly(className)

        if not className:
            return None

        if ' ' in className:
            # Multiple class names passed, do one at a time
            for oneClassName in className.split(' '):
                self.addClass(oneClassName)
            return

        myClassNames = self._classNames

        # Do not allow duplicates
        if className in myClassNames:
            return

        # Regenerate "classNames" and "class" attr.
        #   TODO: Maybe those should be properties?
        myClassNames.append(className)

        return None