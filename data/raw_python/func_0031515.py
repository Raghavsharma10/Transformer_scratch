def lineage(self, title):
        """
        Get lineage information from the taxonomy database for a given title.

        @param title: A C{str} sequence title (e.g., from a BLAST hit). Of the
            form 'gi|63148399|gb|DQ011818.1| Description...'. It is the gi
            number (63148399 in this example) that is looked up in the taxonomy
            database.
        @return: A C{list} of the taxonomic categories of the title. Each list
            element is an (C{int}, C{str}) 2-tuple, giving a taxonomy id and
            a scientific name. The first element in the list will correspond to
            C{title}, and each successive element is the parent of the
            preceeding one. If no taxonomy is found, the returned list will be
            empty.
        """
        if title in self._cache:
            return self._cache[title]

        lineage = []
        gi = int(title.split('|')[1])
        query = 'SELECT taxID from gi_taxid where gi = %d' % gi

        try:
            while True:
                self._cursor.execute(query)
                taxID = self._cursor.fetchone()[0]
                if taxID == 1:
                    break
                query = 'SELECT name from names where taxId = %s' % taxID
                self._cursor.execute(query)
                scientificName = self._cursor.fetchone()[0]
                lineage.append((taxID, scientificName))
                # Move up to the parent.
                query = ('SELECT parent_taxID from nodes where taxID = %s' %
                         taxID)
        except TypeError:
            lineage = []

        self._cache[title] = lineage
        return lineage