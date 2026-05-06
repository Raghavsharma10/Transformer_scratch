def align_and_build_tree(seqs, moltype, best_tree=False, params=None):
    """Returns an alignment and a tree from Sequences object seqs.

    seqs: a cogent.core.alignment.SequenceCollection object, or data that can
    be used to build one.

    moltype: cogent.core.moltype.MolType object

    best_tree: if True (default:False), uses a slower but more accurate
    algorithm to build the tree.

    params: dict of parameters to pass in to the Muscle app controller.

    The result will be a tuple containing a cogent.core.alignment.Alignment
    and a cogent.core.tree.PhyloNode object (or None for the alignment
    and/or tree if either fails).
    """
    aln = align_unaligned_seqs(seqs, moltype=moltype, params=params)
    tree = build_tree_from_alignment(aln, moltype, best_tree, params)
    return {'Align':aln, 'Tree':tree}