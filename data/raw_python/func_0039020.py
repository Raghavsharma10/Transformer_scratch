def run_blast_commands(ncbicommandline_method, **keywords):
    """Runs blastplus/tblastn search, collects result and pass as a xml temporary file.  """

    # temporary files for output
    blast_out_tmp = tempfile.NamedTemporaryFile(mode="w+",delete=False)
    keywords['out'] = blast_out_tmp.name

    # unpack query temp file object
    query_file_object_tmp = keywords['query']
    keywords['query'] = query_file_object_tmp.name

    stderr = ''
    error_string = ''
    try:
        # formating blastplus command
        blastplusx_cline = ncbicommandline_method(**keywords)
        stdout, stderr = blastplusx_cline()

    except ApplicationError as e:
        error_string = "Runtime error: " + stderr + "\n" + e.cmd

    # remove query temp file
    os.unlink(query_file_object_tmp.name)
    # os.remove(query_file_object_tmp.name)

    return blast_out_tmp, error_string