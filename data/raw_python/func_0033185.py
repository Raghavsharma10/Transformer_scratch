def assign_taxonomy(
        data, min_confidence=0.80, output_fp=None, training_data_fp=None,
        fixrank=True, max_memory=None, tmp_dir=tempfile.gettempdir()):
    """Assign taxonomy to each sequence in data with the RDP classifier

        data: open fasta file object or list of fasta lines
        confidence: minimum support threshold to assign taxonomy to a sequence
        output_fp: path to write output; if not provided, result will be
         returned in a dict of {seq_id:(taxonomy_assignment,confidence)}
    """
    # Going to iterate through this twice in succession, best to force
    # evaluation now
    data = list(data)

    # RDP classifier doesn't preserve identifiers with spaces
    # Use lookup table
    seq_id_lookup = {}
    for seq_id, seq in parse_fasta(data):
        seq_id_lookup[seq_id.split()[0]] = seq_id

    app_kwargs = {}
    if tmp_dir is not None:
        app_kwargs['TmpDir'] = tmp_dir
    app = RdpClassifier(**app_kwargs)

    if max_memory is not None:
        app.Parameters['-Xmx'].on(max_memory)

    temp_output_file = tempfile.NamedTemporaryFile(
        prefix='RdpAssignments_', suffix='.txt', dir=tmp_dir)
    app.Parameters['-o'].on(temp_output_file.name)
    if training_data_fp is not None:
        app.Parameters['-t'].on(training_data_fp)

    if fixrank:
        app.Parameters['-f'].on('fixrank')
    else:
        app.Parameters['-f'].on('allrank')

    app_result = app(data)

    assignments = {}

    # ShortSequenceException messages are written to stdout
    # Tag these ID's as unassignable
    for line in app_result['StdOut']:
        excep = parse_rdp_exception(line)
        if excep is not None:
            _, rdp_id = excep
            orig_id = seq_id_lookup[rdp_id]
            assignments[orig_id] = ('Unassignable', 1.0)

    for line in app_result['Assignments']:
        rdp_id, direction, taxa = parse_rdp_assignment(line)
        if taxa[0][0] == "Root":
            taxa = taxa[1:]
        orig_id = seq_id_lookup[rdp_id]
        lineage, confidence = get_rdp_lineage(taxa, min_confidence)
        if lineage:
            assignments[orig_id] = (';'.join(lineage), confidence)
        else:
            assignments[orig_id] = ('Unclassified', 1.0)

    if output_fp:
        try:
            output_file = open(output_fp, 'w')
        except OSError:
            raise OSError("Can't open output file for writing: %s" % output_fp)
        for seq_id, assignment in assignments.items():
            lineage, confidence = assignment
            output_file.write(
                '%s\t%s\t%1.3f\n' % (seq_id, lineage, confidence))
        output_file.close()
        return None
    else:
        return assignments