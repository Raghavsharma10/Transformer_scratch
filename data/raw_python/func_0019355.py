def log(self, sequence, infoarray) -> None:
        """Prepare a |NetCDFFile| object suitable for the given |IOSequence|
        object, when necessary, and pass the given arguments to its
        |NetCDFFile.log| method."""
        if isinstance(sequence, sequencetools.ModelSequence):
            descr = sequence.descr_model
        else:
            descr = 'node'
        if self._isolate:
            descr = '%s_%s' % (descr, sequence.descr_sequence)
            if ((infoarray is not None) and
                    (infoarray.info['type'] != 'unmodified')):
                descr = '%s_%s' % (descr, infoarray.info['type'])
        dirpath = sequence.dirpath_ext
        try:
            files = self.folders[dirpath]
        except KeyError:
            files: Dict[str, 'NetCDFFile'] = collections.OrderedDict()
            self.folders[dirpath] = files
        try:
            file_ = files[descr]
        except KeyError:
            file_ = NetCDFFile(
                name=descr,
                flatten=self._flatten,
                isolate=self._isolate,
                timeaxis=self._timeaxis,
                dirpath=dirpath)
            files[descr] = file_
        file_.log(sequence, infoarray)