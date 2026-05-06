def cmbuild_from_file(stockholm_file_path, refine=False,return_alignment=False,\
    params=None):
    """Uses cmbuild to build a CM file given a stockholm file.

        - stockholm_file_path: a path to a stockholm file.  This file should
            contain a multiple sequence alignment formated in Stockholm format.
            This must contain a sequence structure line:
                #=GC SS_cons <structure string>
        - refine: refine the alignment and realign before building the cm.
            (Default=False)
        - return_alignment: Return alignment and structure string used to
            construct the CM file.  This will either be the original alignment
            and structure string passed in, or the refined alignment if
            --refine was used. (Default=False)
    """
    #get alignment and structure string from stockholm file.
    info, aln, structure_string = \
        list(MinimalRfamParser(open(stockholm_file_path,'U'),\
            seq_constructor=ChangedSequence))[0]

    #call cmbuild_from_alignment.
    res = cmbuild_from_alignment(aln, structure_string, refine=refine, \
        return_alignment=return_alignment,params=params)
    return res