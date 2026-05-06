def _findAssociatedConfigSpecFile(self, cfgFileName):
        """ Given a config file, find its associated config-spec file, and
        return the full pathname of the file. """

        # Handle simplest 2 cases first: co-located or local .cfgspc file
        retval = "."+os.sep+self.__taskName+".cfgspc"
        if os.path.isfile(retval): return retval

        retval = os.path.dirname(cfgFileName)+os.sep+self.__taskName+".cfgspc"
        if os.path.isfile(retval): return retval

        # Also try the resource dir
        retval = self.getDefaultSaveFilename()+'spc' # .cfgspc
        if os.path.isfile(retval): return retval

        # Now try and see if there is a matching .cfgspc file in/under an
        # associated package, if one is defined.
        if self.__assocPkg is not None:
            x, theFile = findCfgFileForPkg(None, '.cfgspc',
                                           pkgObj = self.__assocPkg,
                                           taskName = self.__taskName)
            return theFile

        # Finally try to import the task name and see if there is a .cfgspc
        # file in that directory
        x, theFile = findCfgFileForPkg(self.__taskName, '.cfgspc',
                                       taskName = self.__taskName)
        if os.path.exists(theFile):
            return theFile

        # unfound
        raise NoCfgFileError('Unfound config-spec file for task: "'+ \
                             self.__taskName+'"')