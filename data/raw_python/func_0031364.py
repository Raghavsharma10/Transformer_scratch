def from_genes(cls, genes: List[ExpGene]):
        """Initialize instance using a list of `ExpGene` objects."""
        data = [g.to_dict() for g in genes]
        index = [d.pop('ensembl_id') for d in data]
        table = cls(data, index=index)
        return table