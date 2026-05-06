def _getSaveAsFilter(self):
        """ Return a string to be used as the filter arg to the save file
            dialog during Save-As. """
        # figure the dir to use, start with the one from the file
        absRcDir = os.path.abspath(self._rcDir)
        thedir = os.path.abspath(os.path.dirname(self._taskParsObj.filename))
        # skip if not writeable, or if is _rcDir
        if thedir == absRcDir or not os.access(thedir, os.W_OK):
            thedir = os.path.abspath(os.path.curdir)
        # create save-as filter string
        filt = thedir+'/*.cfg'
        envVarName = APP_NAME.upper()+'_CFG'
        if envVarName in os.environ:
            upx = os.environ[envVarName]
            if len(upx) > 0:  filt = upx+"/*.cfg"
        # done
        return filt