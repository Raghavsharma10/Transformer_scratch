def wf_info(workflow_path):
    """
    Returns the version of the file and the file extension.

    Assumes that the file path is to the file directly ie, ends with a valid file extension.Supports checking local
    files as well as files at http:// and https:// locations. Files at these remote locations are recreated locally to
    enable our approach to version checking, then removed after version is extracted.
    """

    supported_formats = ['py', 'wdl', 'cwl']
    file_type = workflow_path.lower().split('.')[-1]  # Grab the file extension
    workflow_path = workflow_path if ':' in workflow_path else 'file://' + workflow_path

    if file_type in supported_formats:
        if workflow_path.startswith('file://'):
            version = get_version(file_type, workflow_path[7:])
        elif workflow_path.startswith('https://') or workflow_path.startswith('http://'):
            # If file not local go fetch it.
            html = urlopen(workflow_path).read()
            local_loc = os.path.join(os.getcwd(), 'fetchedFromRemote.' + file_type)
            with open(local_loc, 'w') as f:
                f.write(html.decode())
            version = wf_info('file://' + local_loc)[0]  # Don't take the file_type here, found it above.
            os.remove(local_loc)  # TODO: Find a way to avoid recreating file before version determination.
        else:
            raise NotImplementedError('Unsupported workflow file location: {}. Must be local or HTTP(S).'.format(workflow_path))
    else:
        raise TypeError('Unsupported workflow type: .{}. Must be {}.'.format(file_type, '.py, .cwl, or .wdl'))
    return version, file_type.upper()