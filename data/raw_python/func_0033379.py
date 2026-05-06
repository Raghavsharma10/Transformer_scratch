def choices(self):
        """Gets the experiment choices"""

        if self._choices == None:
            self._choices = [ExperimentChoice(self, choice_name) for choice_name in self.choice_names]

        return self._choices