def from_list(cls, l):
        """Generate an GeneSet object from a list of strings.

        Note: See also :meth:`to_list`.

        Parameters
        ----------
        l: list or tuple of str
            A list of strings representing gene set ID, name, genes,
            source, collection, and description. The genes must be
            comma-separated. See also :meth:`to_list`.

        Returns
        -------
        `genometools.basic.GeneSet`
            The gene set.
        """
        id_ = l[0]
        name = l[3]
        genes = l[4].split(',')

        src = l[1] or None
        coll = l[2] or None
        desc = l[5] or None

        return cls(id_, name, genes, src, coll, desc)