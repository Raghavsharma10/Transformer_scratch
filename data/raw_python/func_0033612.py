def align_unaligned_seqs(seqs, moltype=DNA, params=None):
    """Returns an Alignment object from seqs.

    seqs: cogent.core.alignment.SequenceCollection object, or data that can be
    used to build one.

    moltype: a MolType object.  DNA, RNA, or PROTEIN.

    params: dict of parameters to pass in to the Clustal app controller.

    Result will be a cogent.core.alignment.Alignment object.
    """
    #create SequenceCollection object from seqs
    seq_collection = SequenceCollection(seqs,MolType=moltype)
    #Create mapping between abbreviated IDs and full IDs
    int_map, int_keys = seq_collection.getIntMap()
    #Create SequenceCollection from int_map.
    int_map = SequenceCollection(int_map,MolType=moltype)
    #Create Clustalw app.
    app = Clustalw(InputHandler='_input_as_multiline_string',params=params)
    #Get results using int_map as input to app
    res = app(int_map.toFasta())
    #Get alignment as dict out of results
    alignment = dict(ClustalParser(res['Align'].readlines()))
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