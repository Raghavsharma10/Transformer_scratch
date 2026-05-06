def write_zip_fp(fp, data, properties, dir_data_list=None):
    """
        Write custom zip file of data and properties to fp

        :param fp: the file point to which to write the header
        :param data: the data to write to the file; may be None
        :param properties: the properties to write to the file; may be None
        :param dir_data_list: optional list of directory header information structures

        If dir_data_list is specified, data should be None and properties should
        be specified. Then the existing data structure will be left alone and only
        the directory headers and end of directory header will be written.

        Otherwise, if both data and properties are specified, both are written
        out in full.

        The properties param must not change during this method. Callers should
        take care to ensure this does not happen.
    """
    assert data is not None or properties is not None
    # dir_data_list has the format: local file record offset, name, data length, crc32
    dir_data_list = list() if dir_data_list is None else dir_data_list
    dt = datetime.datetime.now()
    if data is not None:
        offset_data = fp.tell()
        def write_data(fp):
            numpy_start_pos = fp.tell()
            numpy.save(fp, data)
            numpy_end_pos = fp.tell()
            fp.seek(numpy_start_pos)
            data_c = numpy.require(data, dtype=data.dtype, requirements=["C_CONTIGUOUS"])
            header_data = fp.read((numpy_end_pos - numpy_start_pos) - data_c.nbytes)  # read the header
            data_crc32 = binascii.crc32(data_c.data, binascii.crc32(header_data)) & 0xFFFFFFFF
            fp.seek(numpy_end_pos)
            return data_crc32
        data_len, crc32 = write_local_file(fp, b"data.npy", write_data, dt)
        dir_data_list.append((offset_data, b"data.npy", data_len, crc32))
    if properties is not None:
        json_str = str()
        try:
            class JSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Geometry.IntPoint) or isinstance(obj, Geometry.IntSize) or isinstance(obj, Geometry.IntRect) or isinstance(obj, Geometry.FloatPoint) or isinstance(obj, Geometry.FloatSize) or isinstance(obj, Geometry.FloatRect):
                        return tuple(obj)
                    else:
                        return json.JSONEncoder.default(self, obj)
            json_io = io.StringIO()
            json.dump(properties, json_io, cls=JSONEncoder)
            json_str = json_io.getvalue()
        except Exception as e:
            # catch exceptions to avoid corrupt zip files
            import traceback
            logging.error("Exception writing zip file %s" + str(e))
            traceback.print_exc()
            traceback.print_stack()
        def write_json(fp):
            json_bytes = bytes(json_str, 'ISO-8859-1')
            fp.write(json_bytes)
            return binascii.crc32(json_bytes) & 0xFFFFFFFF
        offset_json = fp.tell()
        json_len, json_crc32 = write_local_file(fp, b"metadata.json", write_json, dt)
        dir_data_list.append((offset_json, b"metadata.json", json_len, json_crc32))
    dir_offset = fp.tell()
    for offset, name_bytes, data_len, crc32 in dir_data_list:
        write_directory_data(fp, offset, name_bytes, data_len, crc32, dt)
    dir_size = fp.tell() - dir_offset
    write_end_of_directory(fp, dir_size, dir_offset, len(dir_data_list))
    fp.truncate()