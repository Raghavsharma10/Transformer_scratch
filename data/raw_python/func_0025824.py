def updateTitle(self, atitle):
        """ Override so we can append read-only status. """
        if atitle and os.path.exists(atitle):
            if _isInstalled(atitle):
                atitle += '  [installed]'
            elif not os.access(atitle, os.W_OK):
                atitle += '  [read only]'
        super(ConfigObjEparDialog, self).updateTitle(atitle)