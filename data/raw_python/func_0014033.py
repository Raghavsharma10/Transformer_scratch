def get_base_fields(msg, prefix='', parse_header=True):
    '''function to get the full names of every message field in the message'''
    slots = msg.__slots__
    ret_val = []
    msg_types = dict()
    for i in slots:
        slot_msg = getattr(msg, i)
        if not parse_header and i == 'header':
            continue
        if hasattr(slot_msg, '__slots__'):
                (subs, type_map) = get_base_fields(
                    slot_msg, prefix=prefix + i + '.',
                    parse_header=parse_header,
                )

                for i in subs:
                    ret_val.append(i)
                for k, v in type_map.items():
                    msg_types[k] = v
        else:
            ret_val.append(prefix + i)
            msg_types[prefix + i] = slot_msg
    return (ret_val, msg_types)