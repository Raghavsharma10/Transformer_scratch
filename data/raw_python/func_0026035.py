def _setChoiceDict(self):
        """Create dictionary for choice list"""
        # value is name of choice parameter (same as key)
        self.choiceDict = {}
        for c in self.choice: self.choiceDict[c] = c