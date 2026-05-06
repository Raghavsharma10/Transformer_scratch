def sam_parse_reply(line):
    """parse a reply line into a dict"""
    parts = line.split(' ')
    opts = {k: v for (k, v) in split_kv(parts[2:])}
    return SAMReply(parts[0], opts)