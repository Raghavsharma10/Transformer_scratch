def TaskAttemptOutput(output, task_attempt):
    """Returns the correct Output class for a given
    data type, source type, and scatter mode
    """

    (data_type, mode, source_type) = _get_output_info(output)

    if data_type == 'file':
        if mode == 'scatter':
            assert source_type in ['filenames', 'glob'], \
                'source type "%s" not allowed' % source_type
            if source_type == 'filenames':
                return FileListScatterOutput(output, task_attempt)
            return GlobScatterOutput(output, task_attempt)
        else:
            assert mode == 'no_scatter'
            assert source_type == 'filename', \
                'source type "%s" not allowed' % source_type
            return FileOutput(output, task_attempt)
    else:  # data_type is non-file
        if mode == 'scatter':
            assert source_type in [
                'filename', 'filenames', 'glob', 'stream'], \
                'source type "%s" not allowed' % source_type
            if source_type == 'filename':
                return FileContentsScatterOutput(output, task_attempt)
            if source_type == 'filenames':
                return FileListContentsScatterOutput(output, task_attempt)
            if source_type == 'glob':
                return GlobContentsScatterOutput(output, task_attempt)
            assert source_type == 'stream'
            return StreamScatterOutput(output, task_attempt)
        else:
            assert mode == 'no_scatter'
            assert source_type in ['filename', 'stream'], \
                'source type "%s" not allowed' % source_type
            if source_type == 'filename':
                return FileContentsOutput(output, task_attempt)
            assert source_type == 'stream'
            return StreamOutput(output, task_attempt)