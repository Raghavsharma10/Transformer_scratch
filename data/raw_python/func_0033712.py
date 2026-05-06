def add_seqs_to_alignment(seqs, aln, moltype, params=None, accurate=False):
    """Returns an Alignment object from seqs and existing Alignment.

    seqs: a cogent.core.sequence.Sequence object, or data that can be used
    to build one.

    aln: an cogent.core.alignment.Alignment object, or data that can be used
    to build one

    params: dict of parameters to pass in to the Mafft app controller.
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
    app = Mafft(InputHandler='_input_as_multiline_string',\
        params=params,
        SuppressStderr=True)

    #Turn on correct moltype
    moltype_string = moltype.label.upper()
    app.Parameters[MOLTYPE_MAP[moltype_string]].on()

    #Do not report progress
    app.Parameters['--quiet'].on()

    #Add aln_int_map as seed alignment
    app.Parameters['--seed'].on(\
        app._tempfile_as_multiline_string(aln_int_map.toFasta()))

    #More accurate alignment, sacrificing performance.
    if accurate:
        app.Parameters['--globalpair'].on()
        app.Parameters['--maxiterate'].Value=1000

    #Get results using int_map as input to app
    res = app(seq_int_map.toFasta())
    #Get alignment as dict out of results
    alignment = dict(parse_fasta(res['StdOut']))

    #Make new dict mapping original IDs
    new_alignment = {}
    for k,v in alignment.items():
        key = k.replace('_seed_','')
        new_alignment[seq_int_keys[key]]=v
    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment,MolType=moltype)
    #Clean up
    res.cleanUp()
    remove(app.Parameters['--seed'].Value)
    del(seq_collection,seq_int_map,seq_int_keys,\
        aln,aln_int_map,aln_int_keys,app,res,alignment)

    return new_alignment