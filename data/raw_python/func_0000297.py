def checkAndCreateClasses(self, classes):
        """ Function checkAndCreateClasses
        Check and add puppet class

        @param classes: The classes ids list
        @return RETURN: boolean
        """
        actual_classes = self['puppetclasses'].keys()
        for i in classes:
            if i not in actual_classes:
                self['puppetclasses'].append(i)
        self.reload()
        return set(classes).issubset(set((self['puppetclasses'].keys())))