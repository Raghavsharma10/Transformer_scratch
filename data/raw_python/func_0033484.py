def align_unaligned_seqs(seqs, moltype=DNA, params=None):
    """Returns an Alignment object from seqs.

    seqs: SequenceCollection object, or data that can be used to build one.

    moltype: a MolType object.  DNA, RNA, or PROTEIN.

    params: dict of parameters to pass in to the Muscle app controller.

    Result will be an Alignment object.
    """
    if not params:
        params = {}
    #create SequenceCollection object from seqs
    seq_collection = SequenceCollection(seqs,MolType=moltype)
    #Create mapping between abbreviated IDs and full IDs
    int_map, int_keys = seq_collection.getIntMap()
    #Create SequenceCollection from int_map.
    int_map = SequenceCollection(int_map,MolType=moltype)
    #get temporary filename
    params.update({'-out':get_tmp_filename()})
    #Create Muscle app.
    app = Muscle(InputHandler='_input_as_multiline_string',\
                 params=params, WorkingDir=tempfile.gettempdir())
    #Get results using int_map as input to app
    res = app(int_map.toFasta())
    #Get alignment as dict out of results
    alignment = dict(parse_fasta(res['MuscleOut']))
    #Make new dict mapping original IDs
    new_alignment = {}
    for k,v in alignment.items():
        new_alignment[int_keys[k]]=v
    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment,MolType=moltype)
    #Clean up
    res.cleanUp()
    del(seq_collection,int_map,int_keys,app,res,alignment,params)

    return new_alignment