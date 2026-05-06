def annotate(self, fname, tables, feature_strand=False, in_memory=False,
            header=None, out=sys.stdout, parallel=False):
        """
        annotate a file with a number of tables

        Parameters
        ----------

        fname : str or file
           file name or file-handle

        tables : list
            list of tables with which to annotate `fname`

        feature_strand : bool
            if this is True, then the up/downstream designations are based on
            the features in `tables` rather than the features in `fname`

        in_memoory : bool
            if True, then tables are read into memory. This usually makes the
            annotation much faster if there are more than 500 features in
            `fname` and the number of features in the table is less than 100K.

        header : str
            header to print out (if True, use existing header)

        out : file
            where to print output

        parallel : bool
            if True, use multiprocessing library to execute the annotation of
            each chromosome in parallel. Uses more memory.
        """
        from .annotate import annotate
        return annotate(self, fname, tables, feature_strand, in_memory, header=header,
                out=out, parallel=parallel)