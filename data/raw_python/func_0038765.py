def to_csv(self, filename, stimuli=None, inhibitors=None, prepend=""):
        """
        Writes the list of clampings to a CSV file

        Parameters
        ----------
        filename : str
            Absolute path where to write the CSV file

        stimuli : Optional[list[str]]
            List of stimuli names. If given, stimuli are converted to {0,1} instead of {-1,1}.

        inhibitors : Optional[list[str]]
            List of inhibitors names. If given, inhibitors are renamed and converted to {0,1} instead of {-1,1}.

        prepend : str
            Columns are renamed using the given string at the beginning
        """
        self.to_dataframe(stimuli, inhibitors, prepend).to_csv(filename, index=False)