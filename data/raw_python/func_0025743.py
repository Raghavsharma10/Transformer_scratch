def saveParList(self, *args, **kw):
        """Write parameter data to filename (string or filehandle)"""
        if 'filename' in kw:
            filename = kw['filename']
        if not filename:
            filename = self.getFilename()
        if not filename:
            raise ValueError("No filename specified to save parameters")

        if hasattr(filename,'write'):
            fh = filename
            absFileName = os.path.abspath(fh.name)
        else:
            absFileName = os.path.expanduser(filename)
            absDir = os.path.dirname(absFileName)
            if len(absDir) and not os.path.isdir(absDir): os.makedirs(absDir)
            fh = open(absFileName,'w')
        numpars = len(self.__paramList)
        if self._forUseWithEpar: numpars -= 1
        if not self.final_comment: self.final_comment = [''] # force \n at EOF
        # Empty the ConfigObj version of section.defaults since that is based
        # on an assumption incorrect for us, and override with our own list.
        # THIS IS A BIT OF MONKEY-PATCHING!  WATCH FUTURE VERSION CHANGES!
        # See Trac ticket #762.
        while len(self.defaults):
            self.defaults.pop(-1) # empty it, keeping ref
        for key in self._neverWrite:
            self.defaults.append(key)
        # Note also that we are only overwriting the top/main section's
        # "defaults" list, but EVERY [sub-]section has such an attribute...

        # Now write to file, delegating work to ConfigObj (note that ConfigObj
        # write() skips any items listed by name in the self.defaults list)
        self.write(fh)
        fh.close()
        retval = str(numpars) + " parameters written to " + absFileName
        self.filename = absFileName # reset our own ConfigObj filename attr
        self.debug('Keys not written: '+str(self.defaults))
        return retval