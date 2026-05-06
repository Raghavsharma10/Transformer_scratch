def decode_resumable_upload_bitmap(bitmap_node, number_of_units):
    """Decodes bitmap_node to hash of unit_id: is_uploaded

    bitmap_node -- bitmap node of resumable_upload with
                   'count' number and 'words' containing array
    number_of_units -- number of units we are uploading to
                       define the number of bits for bitmap
    """
    bitmap = 0
    for token_id in range(int(bitmap_node['count'])):
        value = int(bitmap_node['words'][token_id])
        bitmap = bitmap | (value << (0xf * token_id))

    result = {}

    for unit_id in range(number_of_units):
        mask = 1 << unit_id
        result[unit_id] = (bitmap & mask) == mask

    return result