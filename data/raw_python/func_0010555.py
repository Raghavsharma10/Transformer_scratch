def bencode(canonical):
    '''
        Turns a dictionary into a bencoded str with alphabetized keys
        e.g., {'spam': 'eggs', 'cow': 'moo'} --> d3:cow3:moo4:spam4:eggse
    '''
    in_dict = dict(canonical)

    def encode_str(in_str):
        out_str = str(len(in_str)) + ':' + in_str
        return out_str

    def encode_int(in_int):
        out_str = str('i' + str(in_int) + 'e')
        return out_str

    def encode_list(in_list):
        out_str = 'l'
        for item in in_list:
            out_str += encode_item(item)
        else:
            out_str += 'e'
        return out_str

    def encode_dict(in_dict):
        out_str = 'd'
        keys = sorted(in_dict.keys())
        for key in keys:
            val = in_dict[key]
            out_str = out_str + encode_item(key) + encode_item(val)
        else:
            out_str += 'e'
        return out_str

    def encode_item(x):
        if isinstance(x, str):
            return encode_str(x)
        elif isinstance(x, int):
            return encode_int(x)
        elif isinstance(x, list):
            return encode_list(x)
        elif isinstance(x, dict):
            return encode_dict(x)

    return encode_item(in_dict)