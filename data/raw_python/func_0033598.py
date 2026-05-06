def cmsearch_from_file(cm_file_path, seqs, moltype, cutoff=0.0, params=None):
    """Uses cmbuild to build a CM file, then cmsearch to find homologs.

        - cm_file_path: path to the file created by cmbuild, containing aligned
            sequences. This will be used to search sequences in seqs.
        - seqs: SequenceCollection object or something that can be used to
            construct one, containing unaligned sequences that are to be
            searched.
        - moltype: cogent.core.moltype object.  Must be DNA or RNA
        - cutoff: bitscore cutoff.  No sequences < cutoff will be kept in
            search results. (Default=0.0).  Infernal documentation suggests
            a cutoff of log2(number nucleotides searching) will give most
            likely true homologs.
    """
    #NOTE: Must degap seqs or Infernal well seg fault!
    seqs = SequenceCollection(seqs,MolType=moltype).degap()
    #Create mapping between abbreviated IDs and full IDs
    int_map, int_keys = seqs.getIntMap()
    #Create SequenceCollection from int_map.
    int_map = SequenceCollection(int_map,MolType=moltype)

    app = Cmsearch(InputHandler='_input_as_paths',WorkingDir='/tmp',\
        params=params)
    app.Parameters['--informat'].on('FASTA')
    app.Parameters['-T'].on(cutoff)

    seqs_path = app._input_as_multiline_string(int_map.toFasta())

    paths = [cm_file_path,seqs_path]

    _, tmp_file = mkstemp(dir=app.WorkingDir)
    app.Parameters['--tabfile'].on(tmp_file)
    res = app(paths)

    search_results = list(CmsearchParser(res['SearchResults'].readlines()))

    if search_results:
        for i,line in enumerate(search_results):
            label = line[1]
            search_results[i][1]=int_keys.get(label,label)

    res.cleanUp()

    return search_results