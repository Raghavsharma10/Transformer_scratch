def raxml_alignment(align_obj,
                 raxml_model="GTRCAT",
                 params={},
                 SuppressStderr=True,
                 SuppressStdout=True):
    """Run raxml on alignment object

    align_obj: Alignment object
    params: you can set any params except -w and -n

    returns: tuple (phylonode,
                    parsimonyphylonode,
                    log likelihood,
                    total exec time)
    """

    # generate temp filename for output
    params["-w"] = "/tmp/"
    params["-n"] = get_tmp_filename().split("/")[-1]
    params["-m"] = raxml_model
    params["-p"] = randint(1,100000)
    ih = '_input_as_multiline_string'
    seqs, align_map = align_obj.toPhylip()

    #print params["-n"]

    # set up command
    raxml_app = Raxml(
                   params=params,
                   InputHandler=ih,
                   WorkingDir=None,
                   SuppressStderr=SuppressStderr,
                   SuppressStdout=SuppressStdout)

    # run raxml
    ra = raxml_app(seqs)

    # generate tree
    tree_node =  DndParser(ra["Result"])

    # generate parsimony tree
    parsimony_tree_node =  DndParser(ra["ParsimonyTree"])

    # extract log likelihood from log file
    log_file = ra["Log"]
    total_exec_time = exec_time = log_likelihood = 0.0
    for line in log_file:
        exec_time, log_likelihood = map(float, line.split())
        total_exec_time += exec_time

    # remove output files
    ra.cleanUp()

    return tree_node, parsimony_tree_node, log_likelihood, total_exec_time