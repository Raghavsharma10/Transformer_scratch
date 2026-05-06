def add_seqs_to_alignment(seqs, aln, params=None):
    """Returns an Alignment object from seqs and existing Alignment.

    seqs: a cogent.core.alignment.SequenceCollection object, or data that can
    be used to build one.

    aln: a cogent.core.alignment.Alignment object, or data that can be used
    to build one

    params: dict of parameters to pass in to the Muscle app controller.
    """
    if not params:
        params = {}

    #create SequenceCollection object from seqs
    seqs_collection = SequenceCollection(seqs)
    #Create mapping between abbreviated IDs and full IDs
    seqs_int_map, seqs_int_keys = seqs_collection.getIntMap(prefix='seq_')
    #Create SequenceCollection from int_map.
    seqs_int_map = SequenceCollection(seqs_int_map)

    #create SequenceCollection object from aln
    aln_collection = SequenceCollection(aln)
    #Create mapping between abbreviated IDs and full IDs
    aln_int_map, aln_int_keys = aln_collection.getIntMap(prefix='aln_')
    #Create SequenceCollection from int_map.
    aln_int_map = SequenceCollection(aln_int_map)

    #set output and profile options
    params.update({'-out':get_tmp_filename(), '-profile':True})

    #save seqs to tmp file
    seqs_filename = get_tmp_filename()
    seqs_out = open(seqs_filename,'w')
    seqs_out.write(seqs_int_map.toFasta())
    seqs_out.close()

    #save aln to tmp file
    aln_filename = get_tmp_filename()
    aln_out = open(aln_filename, 'w')
    aln_out.write(aln_int_map.toFasta())
    aln_out.close()

    #Create Muscle app and get results
    app = Muscle(InputHandler='_input_as_multifile', params=params,
                 WorkingDir=tempfile.gettempdir())
    res = app((aln_filename, seqs_filename))

    #Get alignment as dict out of results
    alignment = dict(parse_fasta(res['MuscleOut']))
    #Make new dict mapping original IDs
    new_alignment = {}
    for k,v in alignment.items():
        if k in seqs_int_keys:
            new_alignment[seqs_int_keys[k]] = v
        else:
            new_alignment[aln_int_keys[k]] = v

    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment)

    #Clean up
    res.cleanUp()
    del(seqs_collection, seqs_int_map, seqs_int_keys)
    del(aln_collection, aln_int_map, aln_int_keys)
    del(app, res, alignment, params)
    remove(seqs_filename)
    remove(aln_filename)

    return new_alignment