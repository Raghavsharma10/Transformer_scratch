def to_json(self, filename):
        """
        Writes the experimental setup to a JSON file

        Parameters
        ----------
        filename : str
            Absolute path where to write the JSON file
        """
        with open(filename, 'w') as fp:
            json.dump(dict(stimuli=self.stimuli, inhibitors=self.inhibitors, readouts=self.readouts), fp)