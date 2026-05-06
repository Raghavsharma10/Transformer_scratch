def _showAnyHelp(self, kind, tag=None):
        """ Invoke task/epar/etc. help and put the page in a window.
        This same logic is used for GUI help, task help, log msgs, etc. """

        # sanity check
        assert kind in ('epar', 'task', 'log'), 'Unknown help kind: '+str(kind)

        #-----------------------------------------
        # See if they'd like to view in a browser
        #-----------------------------------------
        if self._showHelpInBrowser or (kind == 'task' and
                                       self._knowTaskHelpIsHtml):
            if kind == 'epar':
                self.htmlHelp(helpString=self._appHelpString,
                              title='Parameter Editor Help')
            if kind == 'task':
                self.htmlHelp(istask=True, tag=tag)
            if kind == 'log':
                self.htmlHelp(helpString='\n'.join(self._msgHistory),
                              title=self._appName+' Event Log')
            return

        #-----------------------------------------
        # Now try to pop up the regular Tk window
        #-----------------------------------------
        wins = {'epar':self.eparHelpWin,
                'task':self.irafHelpWin,
                'log': self.logHistWin, }
        window = wins[kind]
        try:
            if window.state() != NORMAL:
                window.deiconify()
            window.tkraise()
            return
        except (AttributeError, TclError):
            pass

        #---------------------------------------------------------
        # That didn't succeed (window is still None), so build it
        #---------------------------------------------------------
        if kind == 'epar':
            self.eparHelpWin = self.makeHelpWin(self._appHelpString,
                                                title='Parameter Editor Help')
        if kind == 'task':
            # Acquire the task help as a string
            # Need to include the package name for the task to
            # avoid name conflicts with tasks from other packages. WJH
            self.irafHelpWin = self.makeHelpWin(self.getHelpString(
                                                self.pkgName+'.'+self.taskName))
        if kind == 'log':
            self.logHistWin = self.makeHelpWin('\n'.join(self._msgHistory),
                                               title=self._appName+' Event Log')