def src_process(state, input_data, output_data, ratio, end_of_input=0):
    """Standard processing function.

    Returns non zero on error.
    """
    input_frames, _ = _check_data(input_data)
    output_frames, _ = _check_data(output_data)
    data = ffi.new('SRC_DATA*')
    data.input_frames = input_frames
    data.output_frames = output_frames
    data.src_ratio = ratio
    data.data_in = ffi.cast('float*', ffi.from_buffer(input_data))
    data.data_out = ffi.cast('float*', ffi.from_buffer(output_data))
    data.end_of_input = end_of_input
    error = _lib.src_process(state, data)
    return error, data.input_frames_used, data.output_frames_gen