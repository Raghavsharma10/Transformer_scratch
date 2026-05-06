def _load(self):
        """Load data from a pickle file. """
        with open(self._pickle_file, 'rb') as source:
            pickler = pickle.Unpickler(source)

            for attribute in self._pickle_attributes:
                pickle_data = pickler.load()
                setattr(self, attribute, pickle_data)