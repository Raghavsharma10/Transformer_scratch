def removeClass(self, className):
        '''
            removeClass - remove a class name if present. Returns the class name if  removed, otherwise None.

                @param className <str> - The name of the class to remove

                @return <str> - The class name removed if one was removed, otherwise None if #className wasn't present
        '''
        className = stripWordsOnly(className)

        if not className:
            return None

        if ' ' in className:
            # Multiple class names passed, do one at a time
            for oneClassName in className.split(' '):
                self.removeClass(oneClassName)
            return

        myClassNames = self._classNames

        # If not present, this is a no-op
        if className not in myClassNames:
            return None


        myClassNames.remove(className)

        return className