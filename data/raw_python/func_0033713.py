def align_two_alignments(aln1, aln2, moltype, params=None):
    """Returns an Alignment object from two existing Alignments.

    aln1, aln2: cogent.core.alignment.Alignment objects, or data that can be
    used to build them.
        - Mafft profile alignment only works with aligned sequences. Alignment
        object used to handle unaligned sequences.

    params: dict of parameters to pass in to the Mafft app controller.
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
    app = Mafft(InputHandler='_input_as_paths',\
        params=params,
        SuppressStderr=False)
    app._command = 'mafft-profile'

    aln1_path = app._tempfile_as_multiline_string(aln1_int_map.toFasta())
    aln2_path = app._tempfile_as_multiline_string(aln2_int_map.toFasta())
    filepaths = [aln1_path,aln2_path]

    #Get results using int_map as input to app
    res = app(filepaths)

    #Get alignment as dict out of results
    alignment = dict(parse_fasta(res['StdOut']))

    #Make new dict mapping original IDs
    new_alignment = {}
    for k,v in alignment.items():
        key = k.replace('_seed_','')
        new_alignment[aln1_int_keys[key]]=v
    #Create an Alignment object from alignment dict
    new_alignment = Alignment(new_alignment,MolType=moltype)
    #Clean up
    res.cleanUp()
    remove(aln1_path)
    remove(aln2_path)
    remove('pre')
    remove('trace')
    del(aln1,aln1_int_map,aln1_int_keys,\
        aln2,aln2_int_map,aln2_int_keys,app,res,alignment)

    return new_alignment