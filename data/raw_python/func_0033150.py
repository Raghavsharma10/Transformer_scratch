def _get_result_paths(self, data):
        """ Build the dict of result filepaths
        """
        # get the filepath of the indexed database (after comma)
        # /path/to/refseqs.fasta,/path/to/refseqs.idx
        #                        ^------------------^
        db_name = (self.Parameters['--ref'].Value).split(',')[1]

        result = {}
        extensions = ['bursttrie', 'kmer', 'pos', 'stats']
        for extension in extensions:
            for file_path in glob("%s.%s*" % (db_name, extension)):
                # this will match e.g. nr.bursttrie_0.dat, nr.bursttrie_1.dat
                # and nr.stats
                key = file_path.split(db_name + '.')[1]
                result[key] = ResultPath(Path=file_path, IsWritten=True)
        return result