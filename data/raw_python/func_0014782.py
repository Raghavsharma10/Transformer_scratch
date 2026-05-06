def TaskAttemptInput(input, task_attempt):
    """Returns the correct Input class for a given
    data type and gather mode
    """

    (data_type, mode) = _get_input_info(input)

    if data_type != 'file':
        return NoOpInput(None, task_attempt)

    if mode == 'no_gather':
        return FileInput(input['data']['contents'], task_attempt)
    else:
        assert mode.startswith('gather')
        return FileListInput(input['data']['contents'], task_attempt)