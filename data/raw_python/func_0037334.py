def get_gene_pathways(self, gene_name=None, gene_symbol=None, gene_id=None, pathway_id=None,
                          pathway_name=None, limit=None, as_df=False):
        """Get gene pathway link
        
        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :param str gene_name: gene name 
        :param str gene_symbol: gene symbol
        :param int gene_id: NCBI Gene identifier
        :param pathway_id: 
        :param str pathway_name: pathway name
        :param int limit: maximum number of results
        :return: list of :class:`pyctd.manager.database.models.GenePathway` objects

        .. seealso::
            
            :class:`pyctd.manager.models.GenePathway`

            which is linked to:
            :class:`pyctd.manager.models.Gene`
            :class:`pyctd.manager.models.Pathway`
        """
        q = self.session.query(models.GenePathway)
        q = self._join_gene(q, gene_name=gene_name, gene_symbol=gene_symbol, gene_id=gene_id)
        q = self._join_pathway(q, pathway_id=pathway_id, pathway_name=pathway_name)

        return self._limit_and_df(q, limit, as_df)