def get_goa_gene_sets(go_annotations):
    """Generate a list of gene sets from a collection of GO annotations.

    Each gene set corresponds to all genes annotated with a certain GO term.
    """
    go_term_genes = OrderedDict()
    term_ids = {}
    for ann in go_annotations:
        term_ids[ann.go_term.id] = ann.go_term
        try:
            go_term_genes[ann.go_term.id].append(ann.db_symbol)
        except KeyError:
            go_term_genes[ann.go_term.id] = [ann.db_symbol]
    
    go_term_genes = OrderedDict(sorted(go_term_genes.items()))
    gene_sets = []
    for tid, genes in go_term_genes.items():
        go_term = term_ids[tid]
        gs = GeneSet(id=tid, name=go_term.name, genes=genes,
                     source='GO',
                     collection=go_term.domain_short,
                     description=go_term.definition)
        gene_sets.append(gs)
    gene_sets = GeneSetCollection(gene_sets)
    return gene_sets