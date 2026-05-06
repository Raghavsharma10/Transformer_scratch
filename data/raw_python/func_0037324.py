def get_pathway(self, pathway_name=None, pathway_id=None, limit=None, as_df=False):
        """Get pathway

        .. note::
            Format of pathway_id is KEGG:X* or REACTOME:X* . X* stands for a sequence of digits

        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :param str pathway_name: pathway name
        :param str pathway_id: KEGG or REACTOME identifier
        :param int limit: maximum number of results
        :return: list of :class:`pyctd.manager.models.Pathway` objects

        .. seealso::

            :class:`pyctd.manager.models.Pathway`
        """
        q = self.session.query(models.Pathway)

        if pathway_name:
            q = q.filter(models.Pathway.pathway_name.like(pathway_name))

        if pathway_id:
            q = q.filter(models.Pathway.pathway_id.like(pathway_id))

        return self._limit_and_df(q, limit, as_df)