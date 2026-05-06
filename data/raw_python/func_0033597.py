def cmsearch_from_alignment(aln, structure_string, seqs, moltype, cutoff=0.0,\
    refine=False,params=None):
    """Uses cmbuild to build a CM file, then cmsearch to find homologs.

        - aln: an Alignment object or something that can be used to construct
            one.  All sequences must be the same length.
        - structure_string: vienna structure string representing the consensus
            stucture for the sequences in aln.  Must be the same length as the
            alignment.
        - seqs: SequenceCollection object or something that can be used to
            construct one, containing unaligned sequences that are to be
            searched.
        - moltype: cogent.core.moltype object.  Must be DNA or RNA
        - cutoff: bitscore cutoff.  No sequences < cutoff will be kept in
            search results. (Default=0.0).  Infernal documentation suggests
            a cutoff of log2(number nucleotides searching) will give most
            likely true homologs.
        - refine: refine the alignment and realign before building the cm.
            (Default=False)
    """
    #NOTE: Must degap seqs or Infernal well seg fault!
    seqs = SequenceCollection(seqs,MolType=moltype).degap()
    #Create mapping between abbreviated IDs and full IDs
    int_map, int_keys = seqs.getIntMap()
    #Create SequenceCollection from int_map.
    int_map = SequenceCollection(int_map,MolType=moltype)

    cm_file, aln_file_string = cmbuild_from_alignment(aln, structure_string,\
        refine=refine,return_alignment=True)

    app = Cmsearch(InputHandler='_input_as_paths',WorkingDir='/tmp',\
        params=params)
    app.Parameters['--informat'].on('FASTA')
    app.Parameters['-T'].on(cutoff)

    to_remove = []

    seqs_path = app._input_as_multiline_string(int_map.toFasta())
    cm_path = app._tempfile_as_multiline_string(cm_file)
    paths = [cm_path,seqs_path]
    to_remove.append(cm_path)

    _, tmp_file = mkstemp(dir=app.WorkingDir)
    app.Parameters['--tabfile'].on(tmp_file)
    res = app(paths)

    search_results = list(CmsearchParser(res['SearchResults'].readlines()))
    if search_results:
        for i,line in enumerate(search_results):
            label = line[1]
            search_results[i][1]=int_keys.get(label,label)

    res.cleanUp()
    for f in to_remove:remove(f)

    return search_results