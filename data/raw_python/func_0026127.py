def forceValue(self, newVal, noteEdited=False):
        """Force-set a parameter entry to the given value"""
        if newVal is None:
            newVal = ""
        self.choice.set(newVal)
        if noteEdited:
            self.widgetEdited(val=newVal, skipDups=False)