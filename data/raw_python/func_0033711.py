def align_unaligned_seqs(seqs,moltype=DNA,params=None,accurate=False):
    """Aligns unaligned sequences

    seqs: either list of sequence objects or list of strings
    add_seq_names: boolean. if True, sequence names are inserted in the list
        of sequences. if False, it assumes seqs is a list of lines of some
        proper format that the program can handle
    """
    #create SequenceCollection object from seqs
    seq_collection = SequenceCollection(seqs,MolType=moltype)
    #Create mapping between abbreviated IDs and full IDs
    int_map, int_keys = seq_collection.getIntMap()
    #Create SequenceCollection from int_map.
    int_map = SequenceCollection(int_map,MolType=moltype)
    #Create Mafft app.
    app = Mafft(InputHandler='_input_as_multiline_string',params=params)

    #Turn on correct moltype
    moltype_string = moltype.label.upper()
    app.Parameters[MOLTYPE_MAP[moltype_string]].on()

    #Do not report progress
    app.Parameters['--quiet'].on()

    #More accurate alignment, sacrificing performance.
    if accurate:
        app.Parameters['--globalpair'].on()
        app.Parameters['--maxiterate'].Value=1000

    #Get results using int_map as input to app
    res = app(int_map.toFasta())
    #Get alignment as dict out of results
    alignment = dict(parse_fasta(res['StdOut']))
    #Make new dict mapping original IDs
    new_alignment = {}
    for k,v in alignment.items():
        new_alignment[int_keys[k]]=v
    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment,MolType=moltype)
    #Clean up
    res.cleanUp()
    del(seq_collection,int_map,int_keys,app,res,alignment)

    return new_alignment