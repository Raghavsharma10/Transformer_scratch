def _setChoiceDict(self):
        """Create min-match dictionary for choice list"""
        # value is full name of choice parameter
        self.choiceDict = minmatch.MinMatchDict()
        for c in self.choice: self.choiceDict.add(c, c)