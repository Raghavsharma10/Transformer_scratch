def from_gene_ids_and_names(cls, gene_names: Dict[str, str]):
        """Initialize instance from gene IDs and names."""
        genes = [ExpGene(id_, name=name) for id_, name in gene_names.items()]
        return cls.from_genes(genes)