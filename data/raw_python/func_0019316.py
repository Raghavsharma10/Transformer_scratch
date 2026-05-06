def load_ext(self):
        """Read the internal data from an external data file."""
        try:
            sequencemanager = hydpy.pub.sequencemanager
        except AttributeError:
            raise RuntimeError(
                'The time series of sequence %s cannot be loaded.  Firstly, '
                'you have to prepare `pub.sequencemanager` correctly.'
                % objecttools.devicephrase(self))
        sequencemanager.load_file(self)