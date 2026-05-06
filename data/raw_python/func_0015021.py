def _save(self):
        """Save the attributes defined on _pickle_attributes in a pickle file.

        This improves a lot the nth run as the log file does not need to be
        processed every time.
        """
        with open(self._pickle_file, 'wb') as source:
            pickler = pickle.Pickler(source, pickle.HIGHEST_PROTOCOL)

            for attribute in self._pickle_attributes:
                attr = getattr(self, attribute, None)
                pickler.dump(attr)