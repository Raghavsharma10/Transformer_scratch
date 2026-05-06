def instantiate_from_config(cfg):
    """Instantiate data types from config"""
    for h in cfg:
        if h.get("type") in data_types:
            raise KeyError("Data type '%s' already exists" % h)
        data_types[h.get("type")] = DataType(h)