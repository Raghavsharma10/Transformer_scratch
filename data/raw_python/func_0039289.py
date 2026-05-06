def addBinder(self, binder):
        """Adds a binder to the file
        """
        root = self.etree
        bindings = root.find('bindings')
        bindings.append(binder.etree)

        return True