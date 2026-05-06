def ConnectTo(self, appName, data=None):
        """Exceptional error is handled in zdde Init() method, so the exception
        must be re-raised"""
        global number_of_apps_communicating
        self.ddeServerName = appName
        try:
            self.ddec = DDEClient(self.ddeServerName, self.ddeClientName) # establish conversation
        except DDEError:
            raise
        else:
            number_of_apps_communicating +=1