def cmalign_from_alignment(aln, structure_string, seqs, moltype=DNA,\
    include_aln=True,refine=False, return_stdout=False,params=None,\
    cmbuild_params=None):
    """Uses cmbuild to build a CM file, then cmalign to build an alignment.

        - aln: an Alignment object or something that can be used to construct
            one.  All sequences must be the same length.
        - structure_string: vienna structure string representing the consensus
            stucture for the sequences in aln.  Must be the same length as the
            alignment.
        - seqs: SequenceCollection object or something that can be used to
            construct one, containing unaligned sequences that are to be aligned
            to the aligned sequences in aln.
        - moltype: Cogent moltype object.  Must be RNA or DNA.
        - include_aln: Boolean to include sequences in aln in final alignment.
            (Default=True)
        - refine: refine the alignment and realign before building the cm.
            (Default=False)
        - return_stdout: Boolean to return standard output from infernal.  This
            includes alignment and structure bit scores and average
            probabilities for each sequence. (Default=False)
    """
    #NOTE: Must degap seqs or Infernal well seg fault!
    seqs = SequenceCollection(seqs,MolType=moltype).degap()
    #Create mapping between abbreviated IDs and full IDs
    int_map, int_keys = seqs.getIntMap()
    #Create SequenceCollection from int_map.
    int_map = SequenceCollection(int_map,MolType=moltype)

    cm_file, aln_file_string = cmbuild_from_alignment(aln, structure_string,\
        refine=refine,return_alignment=True,params=cmbuild_params)

    if params is None:
        params = {}
    params.update({MOLTYPE_MAP[moltype]:True})

    app = Cmalign(InputHandler='_input_as_paths',WorkingDir='/tmp',\
        params=params)
    app.Parameters['--informat'].on('FASTA')

    #files to remove that aren't cleaned up by ResultPath object
    to_remove = []
    #turn on --withali flag if True.
    if include_aln:
        app.Parameters['--withali'].on(\
            app._tempfile_as_multiline_string(aln_file_string))
        #remove this file at end
        to_remove.append(app.Parameters['--withali'].Value)

    seqs_path = app._input_as_multiline_string(int_map.toFasta())
    cm_path = app._tempfile_as_multiline_string(cm_file)

    #add cm_path to to_remove
    to_remove.append(cm_path)
    paths = [cm_path,seqs_path]

    _, tmp_file = mkstemp(dir=app.WorkingDir)
    app.Parameters['-o'].on(tmp_file)

    res = app(paths)

    info, aligned, struct_string = \
        list(MinimalRfamParser(res['Alignment'].readlines(),\
            seq_constructor=SEQ_CONSTRUCTOR_MAP[moltype]))[0]

    #Make new dict mapping original IDs
    new_alignment={}
    for k,v in aligned.NamedSeqs.items():
        new_alignment[int_keys.get(k,k)]=v
    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment,MolType=moltype)

    std_out = res['StdOut'].read()
    #clean up files
    res.cleanUp()
    for f in to_remove: remove(f)

    if return_stdout:
        return new_alignment, struct_string, std_out
    else:
        return new_alignment, struct_string