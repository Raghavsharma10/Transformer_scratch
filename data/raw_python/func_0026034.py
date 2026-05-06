def _setChoice(self,s,strict=0):
        """Set choice parameter from string s"""
        clist = _getChoice(s,strict)
        self.choice = list(map(self._coerceValue, clist))
        self._setChoiceDict()