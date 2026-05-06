def src_simple(input_data, output_data, ratio, converter_type, channels):
    """Perform a single conversion from an input buffer to an output buffer.

    Simple interface for performing a single conversion from input buffer to
    output buffer at a fixed conversion ratio. Simple interface does not require
    initialisation as it can only operate on a single buffer worth of audio.
    """
    input_frames, _ = _check_data(input_data)
    output_frames, _ = _check_data(output_data)
    data = ffi.new('SRC_DATA*')
    data.input_frames = input_frames
    data.output_frames = output_frames
    data.src_ratio = ratio
    data.data_in = ffi.cast('float*', ffi.from_buffer(input_data))
    data.data_out = ffi.cast('float*', ffi.from_buffer(output_data))
    error = _lib.src_simple(data, converter_type, channels)
    return error, data.input_frames_used, data.output_frames_gen