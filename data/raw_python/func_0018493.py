def bin_query(self, table, chrom, start, end):
        """
        perform an efficient spatial query using the bin column if available.
        The possible bins are calculated from the `start` and `end` sent to
        this function.

        Parameters
        ----------

        table : str or table
           table to query

        chrom : str
           chromosome for the query

        start : int
           0-based start postion

        end : int
            0-based end position

        """
        if isinstance(table, six.string_types):
            table = getattr(self, table)

        try:
            tbl = table._table
        except AttributeError:
            tbl = table.column_descriptions[0]['type']._table

        q = table.filter(tbl.c.chrom == chrom)

        if hasattr(tbl.c, "bin"):
            bins = Genome.bins(start, end)
            if len(bins) < 100:
                q = q.filter(tbl.c.bin.in_(bins))

        if hasattr(tbl.c, "txStart"):
            return q.filter(tbl.c.txStart <= end).filter(tbl.c.txEnd >= start)
        return q.filter(tbl.c.chromStart <= end).filter(tbl.c.chromEnd >= start)