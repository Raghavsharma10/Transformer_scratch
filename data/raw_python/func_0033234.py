def insert_sequences_into_tree(aln, moltype, params={}):
    """Returns a tree from placement of sequences
    """
    # convert aln to phy since seq_names need fixed to run through parsinsert
    new_aln=get_align_for_phylip(StringIO(aln))

    # convert aln to fasta in case it is not already a fasta file
    aln2 = Alignment(new_aln)
    seqs = aln2.toFasta()

    parsinsert_app = ParsInsert(params=params)
    result = parsinsert_app(seqs)

    # parse tree
    tree = DndParser(result['Tree'].read(), constructor=PhyloNode)

    # cleanup files
    result.cleanUp()

    return tree