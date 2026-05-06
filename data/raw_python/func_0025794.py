def checkAllTriggers(self, action):
        """ Go over all widgets and let them know they have been edited
            recently and they need to check for any trigger actions.  This
            would be used right after all the widgets have their values
            set or forced (e.g. via setAllEntriesFromParList). """
        for entry in self.entryNo:
            entry.widgetEdited(action=action, skipDups=False)