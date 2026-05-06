def taxTree(taxdict):
    """Return taxonomic Newick tree"""
    # the taxonomic dictionary holds the lineage of each ident in
    #  the same order as the taxonomy
    # use hierarchy to construct a taxonomic tree
    for rank in taxdict.taxonomy:
        current_level = float(taxdict.taxonomy.index(rank))
        # get clades at this rank in hierarchy
        clades = taxdict.hierarchy[rank]
        # merge those that are in the same clade into a cladestring
        for clade in clades:
            # unpack the identities in this clade and its clade name
            cladeidents, cladename = clade
            # Remove '' TaxRefs -- in cladestring already
            cladeidents = [e for e in cladeidents if e.ident]
            # only create cladestring if more than one ident in clade
            if len(cladeidents) < 2:
                continue
            # label node by 'clade'_'rank'
            cladename = '{0}_{1}'.format(cladename, rank)
            cladestring = stringClade(cladeidents, cladename, current_level)
            # replace first TaxRef in cladeidents with cladestring
            cladeidents[0].change(ident=cladestring, rank=rank)
            # replace all other TaxRefs with ''
            for e in cladeidents[1:]:
                e.change(ident='', rank=rank)
    # join any remaining strands into tree
    if len(taxdict.hierarchy[taxdict.taxonomy[-1]]) > 1:
        # unlist first
        clade = [e[0] for e in taxdict.hierarchy[taxdict.taxonomy[-1]]]
        cladeidents = sum(clade, [])
        cladeidents = [e for e in cladeidents if e.ident]
        cladestring = stringClade(cladeidents, 'life', current_level+1)
    return cladestring + ';'