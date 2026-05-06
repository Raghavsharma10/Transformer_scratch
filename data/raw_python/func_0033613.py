def add_seqs_to_alignment(seqs, aln, moltype, params=None):
    """Returns an Alignment object from seqs and existing Alignment.

    seqs: a cogent.core.alignment.SequenceCollection object, or data that can
    be used to build one.

    aln: a cogent.core.alignment.Alignment object, or data that can be used to
    build one

    params: dict of parameters to pass in to the Clustal app controller.
    """
    #create SequenceCollection object from seqs
    seq_collection = SequenceCollection(seqs,MolType=moltype)
    #Create mapping between abbreviated IDs and full IDs
    seq_int_map, seq_int_keys = seq_collection.getIntMap()
    #Create SequenceCollection from int_map.
    seq_int_map = SequenceCollection(seq_int_map,MolType=moltype)

    #create Alignment object from aln
    aln = Alignment(aln,MolType=moltype)
    #Create mapping between abbreviated IDs and full IDs
    aln_int_map, aln_int_keys = aln.getIntMap(prefix='seqn_')
    #Create SequenceCollection from int_map.
    aln_int_map = Alignment(aln_int_map,MolType=moltype)

    #Update seq_int_keys with aln_int_keys
    seq_int_keys.update(aln_int_keys)

    #Create Mafft app.
    app = Clustalw(InputHandler='_input_as_multiline_string',\
        params=params,
        SuppressStderr=True)
    app.Parameters['-align'].off()
    app.Parameters['-infile'].off()
    app.Parameters['-sequences'].on()

    #Add aln_int_map as profile1
    app.Parameters['-profile1'].on(\
        app._tempfile_as_multiline_string(aln_int_map.toFasta()))

    #Add seq_int_map as profile2
    app.Parameters['-profile2'].on(\
        app._tempfile_as_multiline_string(seq_int_map.toFasta()))
    #Get results using int_map as input to app
    res = app()

    #Get alignment as dict out of results
    alignment = dict(ClustalParser(res['Align'].readlines()))

    #Make new dict mapping original IDs
    new_alignment = {}
    for k,v in alignment.items():
        new_alignment[seq_int_keys[k]]=v
    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment,MolType=moltype)
    #Clean up
    res.cleanUp()
    remove(app.Parameters['-profile1'].Value)
    remove(app.Parameters['-profile2'].Value)
    del(seq_collection,seq_int_map,seq_int_keys,\
        aln,aln_int_map,aln_int_keys,app,res,alignment)

    return new_alignment