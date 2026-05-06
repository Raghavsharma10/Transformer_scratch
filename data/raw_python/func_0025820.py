def _overrideMasterSettings(self):
        """ Override so that we can run in a different mode. """
        # config-obj dict of defaults
        cod = self._getGuiSettings()

        # our own GUI setup
        self._appName              = APP_NAME
        self._appHelpString        = tealHelpString
        self._useSimpleAutoClose   = self._do_usac
        self._showExtraHelpButton  = False
        self._saveAndCloseOnExec   = cfgGetBool(cod, 'saveAndCloseOnExec', True)
        self._showHelpInBrowser    = cfgGetBool(cod, 'showHelpInBrowser', False)
        self._writeProtectOnSaveAs = cfgGetBool(cod, 'writeProtectOnSaveAsOpt', True)
        self._flagNonDefaultVals   = cfgGetBool(cod, 'flagNonDefaultVals', None)
        self._optFile              = APP_NAME.lower()+".optionDB"

        # our own colors
        # prmdrss teal: #00ffaa, pure cyan (teal) #00ffff (darker) #008080
        # "#aaaaee" is a darker but good blue, but "#bbbbff" pops
        ltblu = "#ccccff" # light blue
        drktl = "#008888" # darkish teal
        self._frmeColor = cod.get('frameColor', drktl)
        self._taskColor = cod.get('taskBoxColor', ltblu)
        self._bboxColor = cod.get('buttonBoxColor', ltblu)
        self._entsColor = cod.get('entriesColor', ltblu)
        self._flagColor = cod.get('flaggedColor', 'brown')

        # double check _canExecute, but only if it is still set to the default
        if self._canExecute and self._taskParsObj: # default _canExecute=True
            self._canExecute = self._taskParsObj.canExecute()
        self._showExecuteButton = self._canExecute

        # check on the help string - just to see if it is HTML
        # (could use HTMLParser here if need be, be quick and simple tho)
        hhh = self.getHelpString(self.pkgName+'.'+self.taskName)
        if hhh:
            hhh = hhh.lower()
            if hhh.find('<html') >= 0 or hhh.find('</html>') > 0:
                self._knowTaskHelpIsHtml = True
            elif hhh.startswith('http:') or hhh.startswith('https:'):
                self._knowTaskHelpIsHtml = True
            elif hhh.startswith('file:') and \
                 (hhh.endswith('.htm') or hhh.endswith('.html')):
                self._knowTaskHelpIsHtml = True