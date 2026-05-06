def _pack_asf_image(mime, data, type=3, description=""):
    """Pack image data for a WM/Picture tag.
    """
    tag_data = struct.pack('<bi', type, len(data))
    tag_data += mime.encode("utf-16-le") + b'\x00\x00'
    tag_data += description.encode("utf-16-le") + b'\x00\x00'
    tag_data += data
    return tag_data