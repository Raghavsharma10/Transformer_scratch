def fasta_cmd_get_seqs(acc_list,
                 blast_db=None,
                 is_protein=None,
                 out_filename=None,
                 params={},
                 WorkingDir=tempfile.gettempdir(),
                 SuppressStderr=None,
                 SuppressStdout=None):
    """Retrieve sequences for list of accessions """

    if is_protein is None:
        params["-p"] = 'G'
    elif is_protein:
        params["-p"] = 'T'
    else:
        params["-p"] = 'F'

    if blast_db:
        params["-d"] = blast_db

    if out_filename:
        params["-o"] = out_filename

    # turn off duplicate accessions
    params["-a"] = "F"

    # create Psi-BLAST
    fasta_cmd = FastaCmd(params=params,
                       InputHandler='_input_as_string',
                       WorkingDir=WorkingDir,
                       SuppressStderr=SuppressStderr,
                       SuppressStdout=SuppressStdout)

    # return results
    return fasta_cmd("\"%s\"" % ','.join(acc_list))