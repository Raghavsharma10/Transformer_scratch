def from_json(cls, filename):
        """
        Creates an experimental setup from a JSON file

        Parameters
        ----------
        filename : str
            Absolute path to JSON file

        Returns
        -------
        caspo.core.setup.Setup
            Created object instance
        """
        with open(filename) as fp:
            raw = json.load(fp)

        return cls(raw['stimuli'], raw['inhibitors'], raw['readouts'])