def get_gene(self, gene_name=None, gene_symbol=None, gene_id=None, synonym=None, uniprot_id=None,
                 pharmgkb_id=None, biogrid_id=None, alt_gene_id=None, limit=None, as_df=False):
        """Get genes

        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :param alt_gene_id: 
        :param str gene_name: gene name
        :param str gene_symbol: HGNC gene symbol
        :param int gene_id: NCBI Entrez Gene identifier
        :param str synonym: Synonym
        :param str uniprot_id: UniProt primary accession number
        :param str pharmgkb_id: PharmGKB identifier 
        :param int biogrid_id: BioGRID identifier
        :param int limit: maximum of results 
        :rtype: list[models.Gene]
        """
        q = self.session.query(models.Gene)

        if gene_symbol:
            q = q.filter(models.Gene.gene_symbol.like(gene_symbol))

        if gene_name:
            q = q.filter(models.Gene.gene_name.like(gene_name))

        if gene_id:
            q = q.filter(models.Gene.gene_id.like(gene_id))

        if synonym:
            q = q.join(models.GeneSynonym).filter(models.GeneSynonym.synonym == synonym)

        if uniprot_id:
            q = q.join(models.GeneUniprot).filter(models.GeneUniprot.uniprot_id == uniprot_id)

        if pharmgkb_id:
            q = q.join(models.GenePharmgkb).filter(models.GenePharmgkb.pharmgkb_id == pharmgkb_id)

        if biogrid_id:
            q = q.join(models.GeneBiogrid).filter(models.GeneBiogrid.biogrid_id == biogrid_id)

        if alt_gene_id:
            q = q.join(models.GeneAltGeneId.alt_gene_id == alt_gene_id)

        return self._limit_and_df(q, limit, as_df)