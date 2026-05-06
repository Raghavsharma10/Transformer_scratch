def _recSortKey(r):
        """
        Sort order for Features, by genomic coordinate,
        disambiguated by feature type (alphabetically).
        """
        return r.seqname, r.start, -r.end, r.type