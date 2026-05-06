def save_ext(self):
        """Write the internal data into an external data file."""
        try:
            sequencemanager = hydpy.pub.sequencemanager
        except AttributeError:
            raise RuntimeError(
                'The time series of sequence %s cannot be saved.  Firstly,'
                'you have to prepare `pub.sequencemanager` correctly.'
                % objecttools.devicephrase(self))
        sequencemanager.save_file(self)