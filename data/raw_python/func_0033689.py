def mothur_classify_file(
        query_file, ref_fp, tax_fp, cutoff=None, iters=None, ksize=None,
        output_fp=None, tmp_dir=None):
    """Classify a set of sequences using Mothur's naive bayes method

    Dashes are used in Mothur to provide multiple filenames.  A
    filepath with a dash typically breaks an otherwise valid command
    in Mothur.  This wrapper script makes a copy of both files, ref_fp
    and tax_fp, to ensure that the path has no dashes.

    For convenience, we also ensure that each taxon list in the
    id-to-taxonomy file ends with a semicolon.
    """
    if tmp_dir is None:
        tmp_dir = gettempdir()

    ref_seq_ids = set()

    user_ref_file = open(ref_fp)
    tmp_ref_file = NamedTemporaryFile(dir=tmp_dir, suffix=".ref.fa")
    for seq_id, seq in parse_fasta(user_ref_file):
        id_token = seq_id.split()[0]
        ref_seq_ids.add(id_token)
        tmp_ref_file.write(">%s\n%s\n" % (seq_id, seq))
    tmp_ref_file.seek(0)

    user_tax_file = open(tax_fp)
    tmp_tax_file = NamedTemporaryFile(dir=tmp_dir, suffix=".tax.txt")
    for line in user_tax_file:
        line = line.rstrip()
        if not line:
            continue

        # MOTHUR is particular that each assignment end with a semicolon.
        if not line.endswith(";"):
            line = line + ";"

        id_token, _, _ = line.partition("\t")
        if id_token in ref_seq_ids:
            tmp_tax_file.write(line)
            tmp_tax_file.write("\n")
    tmp_tax_file.seek(0)

    params = {"reference": tmp_ref_file.name, "taxonomy": tmp_tax_file.name}
    if cutoff is not None:
        params["cutoff"] = cutoff
    if ksize is not None:
        params["ksize"] = ksize
    if iters is not None:
        params["iters"] = iters

    # Create a temporary working directory to accommodate mothur's output
    # files, which are generated automatically based on the input
    # file.
    work_dir = mkdtemp(dir=tmp_dir)

    app = MothurClassifySeqs(
        params, InputHandler='_input_as_lines', WorkingDir=work_dir,
        TmpDir=tmp_dir)
    result = app(query_file)

    # Force evaluation so we can safely clean up files
    assignments = list(parse_mothur_assignments(result['assignments']))
    result.cleanUp()
    rmtree(work_dir)

    if output_fp is not None:
        f = open(output_fp, "w")
        for query_id, taxa, conf in assignments:
            taxa_str = ";".join(taxa)
            f.write("%s\t%s\t%.2f\n" % (query_id, taxa_str, conf))
        f.close()
        return None
    return dict((a, (b, c)) for a, b, c in assignments)