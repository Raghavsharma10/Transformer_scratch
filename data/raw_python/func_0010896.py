def write_mmtf(file_path, input_data, input_function):
    """API function to write data as MMTF to a file

    :param file_path the path of the file to write
    :param input_data the input data in any user format
    :param input_function a function to converte input_data to an output format. Must contain all methods in TemplateEncoder
    """
    mmtf_encoder = MMTFEncoder()
    pass_data_on(input_data, input_function, mmtf_encoder)
    mmtf_encoder.write_file(file_path)