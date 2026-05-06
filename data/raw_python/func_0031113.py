def write_tsv(self, path):
        """Write the database to a tab-delimited text file.

        Parameters
        ----------
        path: str
            The path name of the file.

        Returns
        -------
        None
        """
        with open(path, 'wb') as ofh:
            writer = csv.writer(
                ofh, dialect='excel-tab',
                quoting=csv.QUOTE_NONE, lineterminator=os.linesep
            )
            for gs in self._gene_sets.values():
                writer.writerow(gs.to_list())