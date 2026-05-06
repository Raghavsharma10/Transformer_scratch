def align_two_alignments(aln1, aln2, moltype, params=None):
    """Returns an Alignment object from two existing Alignments.

    aln1, aln2: cogent.core.alignment.Alignment objects, or data that can be
    used to build them.

    params: dict of parameters to pass in to the Clustal app controller.
    """
    #create SequenceCollection object from seqs
    aln1 = Alignment(aln1,MolType=moltype)
    #Create mapping between abbreviated IDs and full IDs
    aln1_int_map, aln1_int_keys = aln1.getIntMap()
    #Create SequenceCollection from int_map.
    aln1_int_map = Alignment(aln1_int_map,MolType=moltype)

    #create Alignment object from aln
    aln2 = Alignment(aln2,MolType=moltype)
    #Create mapping between abbreviated IDs and full IDs
    aln2_int_map, aln2_int_keys = aln2.getIntMap(prefix='seqn_')
    #Create SequenceCollection from int_map.
    aln2_int_map = Alignment(aln2_int_map,MolType=moltype)

    #Update aln1_int_keys with aln2_int_keys
    aln1_int_keys.update(aln2_int_keys)

    #Create Mafft app.
    app = Clustalw(InputHandler='_input_as_multiline_string',\
        params=params,
        SuppressStderr=True)
    app.Parameters['-align'].off()
    app.Parameters['-infile'].off()
    app.Parameters['-profile'].on()

    #Add aln_int_map as profile1
    app.Parameters['-profile1'].on(\
        app._tempfile_as_multiline_string(aln1_int_map.toFasta()))

    #Add seq_int_map as profile2
    app.Parameters['-profile2'].on(\
        app._tempfile_as_multiline_string(aln2_int_map.toFasta()))
    #Get results using int_map as input to app
    res = app()

    #Get alignment as dict out of results
    alignment = dict(ClustalParser(res['Align'].readlines()))

    #Make new dict mapping original IDs
    new_alignment = {}
    for k,v in alignment.items():
        new_alignment[aln1_int_keys[k]]=v
    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment,MolType=moltype)
    #Clean up
    res.cleanUp()
    remove(app.Parameters['-profile1'].Value)
    remove(app.Parameters['-profile2'].Value)
    del(aln1,aln1_int_map,aln1_int_keys,\
        aln2,aln2_int_map,aln2_int_keys,app,res,alignment)

    return new_alignment