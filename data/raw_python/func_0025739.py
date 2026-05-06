def getDefaultSaveFilename(self, stub=False):
        """ Return name of file where we are expected to be saved if no files
        for this task have ever been saved, and the user wishes to save.  If
        stub is True, the result will be <dir>/<taskname>_stub.cfg instead of
        <dir>/<taskname>.cfg. """
        if stub:
            return self._rcDir+os.sep+self.__taskName+'_stub.cfg'
        else:
            return self._rcDir+os.sep+self.__taskName+'.cfg'