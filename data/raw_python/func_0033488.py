def align_two_alignments(aln1, aln2, params=None):
    """Returns an Alignment object from two existing Alignments.

    aln1, aln2: cogent.core.alignment.Alignment objects, or data that can be
    used to build them.

    params: dict of parameters to pass in to the Muscle app controller.
    """
    if not params:
        params = {}

    #create SequenceCollection object from aln1
    aln1_collection = SequenceCollection(aln1)
    #Create mapping between abbreviated IDs and full IDs
    aln1_int_map, aln1_int_keys = aln1_collection.getIntMap(prefix='aln1_')
    #Create SequenceCollection from int_map.
    aln1_int_map = SequenceCollection(aln1_int_map)

    #create SequenceCollection object from aln2
    aln2_collection = SequenceCollection(aln2)
    #Create mapping between abbreviated IDs and full IDs
    aln2_int_map, aln2_int_keys = aln2_collection.getIntMap(prefix='aln2_')
    #Create SequenceCollection from int_map.
    aln2_int_map = SequenceCollection(aln2_int_map)

    #set output and profile options
    params.update({'-out':get_tmp_filename(), '-profile':True})

    #save aln1 to tmp file
    aln1_filename = get_tmp_filename()
    aln1_out = open(aln1_filename,'w')
    aln1_out.write(aln1_int_map.toFasta())
    aln1_out.close()

    #save aln2 to tmp file
    aln2_filename = get_tmp_filename()
    aln2_out = open(aln2_filename, 'w')
    aln2_out.write(aln2_int_map.toFasta())
    aln2_out.close()

    #Create Muscle app and get results
    app = Muscle(InputHandler='_input_as_multifile', params=params,
                 WorkingDir=tempfile.gettempdir())
    res = app((aln1_filename, aln2_filename))

    #Get alignment as dict out of results
    alignment = dict(parse_fasta(res['MuscleOut']))

    #Make new dict mapping original IDs
    new_alignment = {}
    for k,v in alignment.items():
        if k in aln1_int_keys:
            new_alignment[aln1_int_keys[k]] = v
        else:
            new_alignment[aln2_int_keys[k]] = v

    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment)

    #Clean up
    res.cleanUp()
    del(aln1_collection, aln1_int_map, aln1_int_keys)
    del(aln2_collection, aln2_int_map, aln2_int_keys)
    del(app, res, alignment, params)
    remove(aln1_filename)
    remove(aln2_filename)

    return new_alignment