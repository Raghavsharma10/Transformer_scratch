def _getOpenChoices(self):
        """ Go through all possible sites to find applicable .cfg files.
            Return as an iterable. """
        tsk = self._taskParsObj.getName()
        taskFiles = set()
        dirsSoFar = [] # this helps speed this up (skip unneeded globs)

        # last dir
        aDir = os.path.dirname(self._taskParsObj.filename)
        if len(aDir) < 1: aDir = os.curdir
        dirsSoFar.append(aDir)
        taskFiles.update(cfgpars.getCfgFilesInDirForTask(aDir, tsk))

        # current dir
        aDir = os.getcwd()
        if aDir not in dirsSoFar:
            dirsSoFar.append(aDir)
            taskFiles.update(cfgpars.getCfgFilesInDirForTask(aDir, tsk))

        # task's python pkg dir (if tsk == python pkg name)
        try:
            x, pkgf = cfgpars.findCfgFileForPkg(tsk, '.cfg', taskName=tsk,
                              pkgObj=self._taskParsObj.getAssocPkg())
            taskFiles.update( (pkgf,) )
        except cfgpars.NoCfgFileError:
            pass # no big deal - maybe there is no python package

        # user's own resourceDir
        aDir = self._rcDir
        if aDir not in dirsSoFar:
            dirsSoFar.append(aDir)
            taskFiles.update(cfgpars.getCfgFilesInDirForTask(aDir, tsk))

        # extra loc - see if they used the app's env. var
        aDir = dirsSoFar[0] # flag to skip this if no env var found
        envVarName = APP_NAME.upper()+'_CFG'
        if envVarName in os.environ: aDir = os.environ[envVarName]
        if aDir not in dirsSoFar:
            dirsSoFar.append(aDir)
            taskFiles.update(cfgpars.getCfgFilesInDirForTask(aDir, tsk))

        # At the very end, add an option which we will later interpret to mean
        # to open the file dialog.
        taskFiles = list(taskFiles) # so as to keep next item at end of seq
        taskFiles.sort()
        taskFiles.append("Other ...")

        return taskFiles