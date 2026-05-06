def deserialize(to_deserialize, *args, **kwargs):
    """
    Deserializes a string into a PyMongo BSON
    """
    if isinstance(to_deserialize, string_types):
        if re.match('^[0-9a-f]{24}$', to_deserialize):
            return ObjectId(to_deserialize)
        try:
            return bson_loads(to_deserialize, *args, **kwargs)
        except:
            return bson_loads(bson_dumps(to_deserialize), *args, **kwargs)
    else:
        return bson_loads(bson_dumps(to_deserialize), *args, **kwargs)