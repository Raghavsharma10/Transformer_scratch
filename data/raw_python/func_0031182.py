def iter(self):
        """
        Iterate over the sequences in the files in self.files_, yielding each
        as an instance of the desired read class.
        """
        for _file in self._files:
            with asHandle(_file) as fp:
                # Use FastqGeneralIterator because it provides access to
                # the unconverted quality string (i.e., it doesn't try to
                # figure out the numeric quality values, which we don't
                # care about at this point).
                for sequenceId, sequence, quality in FastqGeneralIterator(fp):
                    yield self.readClass(sequenceId, sequence, quality)