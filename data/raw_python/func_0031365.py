def from_gene_ids(cls, gene_ids: List[str]):
        """Initialize instance from gene IDs."""
        genes = [ExpGene(id_) for id_ in gene_ids]
        return cls.from_genes(genes)