def _coerceSingleRepetition(self, dataSet):
        """
        Make a new liveform with our parameters, and get it to coerce our data
        for us.
        """
        # make a liveform because there is some logic in _coerced
        form = LiveForm(lambda **k: None, self.parameters, self.name)
        return form.fromInputs(dataSet)