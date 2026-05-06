def getDefaultParList(self):
        """ Return a par list just like ours, but with all default values. """
        # The code below (create a new set-to-dflts obj) is correct, but it
        # adds a tenth of a second to startup.  Clicking "Defaults" in the
        # GUI does not call this.  But this can be used to set the order seen.

        # But first check for rare case of no cfg file name
        if self.filename is None:
            # this is a .cfgspc-only kind of object so far
            self.filename = self.getDefaultSaveFilename(stub=True)
            return copy.deepcopy(self.__paramList)

        tmpObj = ConfigObjPars(self.filename, associatedPkg=self.__assocPkg,
                               setAllToDefaults=True, strict=False)
        return tmpObj.getParList()