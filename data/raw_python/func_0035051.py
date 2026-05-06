def simulateAlignment(model, treeFile, alignmentPrefix, randomSeed=False):
    """
    Simulate an alignment given a model and tree (units = subs/site).

    Simulations done using `pyvolve`.

    Args:
        `model` (`phydmslib.models.Models` object)
            The model used for the simulations. Only
            models that can be passed to `pyvolve.Partitions`
            are supported.
        `treeFile` (str)
            Name of newick file used to simulate the sequences.
            The branch lengths should be in substitutions per site,
            which is the default units for all `phydms` outputs.
        `alignmentPrefix`
            Prefix for the files created by `pyvolve`.

    The result of this function is a simulated FASTA alignment
    file with the name having the prefix giving by `alignmentPrefix`
    and the suffix `'_simulatedalignment.fasta'`.
    """
    if randomSeed == False:
        pass
    else:
        random.seed(randomSeed)

    #Transform the branch lengths by dividing by the model `branchScale`
    tree = Bio.Phylo.read(treeFile, 'newick')
    for node in tree.get_terminals() + tree.get_nonterminals():
        if (node.branch_length == None) and (node == tree.root):
            node.branch_length = 1e-06
        else:
            node.branch_length /= model.branchScale
    fd, temp_path = mkstemp()
    Bio.Phylo.write(tree, temp_path, 'newick')
    os.close(fd)
    pyvolve_tree = pyvolve.read_tree(file=temp_path)
    os.remove(temp_path)


    #Make the `pyvolve` partition
    partitions = pyvolvePartitions(model)

    #Simulate the alignment
    alignment = '{0}_simulatedalignment.fasta'.format(alignmentPrefix)
    info = '_temp_{0}info.txt'.format(alignmentPrefix)
    rates = '_temp_{0}_ratefile.txt'.format(alignmentPrefix)
    evolver = pyvolve.Evolver(partitions=partitions, tree=pyvolve_tree)
    evolver(seqfile=alignment, infofile=info, ratefile=rates)
    for f in [rates,info, "custom_matrix_frequencies.txt"]:
        if os.path.isfile(f):
            os.remove(f)
    assert os.path.isfile(alignment)