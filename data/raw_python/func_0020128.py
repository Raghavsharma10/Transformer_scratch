def pairs_to_dict(response, encoding):
    "Create a dict given a list of key/value pairs"
    it = iter(response)
    return dict(((k.decode(encoding), v) for k, v in zip(it, it)))