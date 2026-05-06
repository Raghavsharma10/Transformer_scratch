def parse_json(json_string, object_type, mappers):
    """
    This function will use the custom JsonDecoder and the conventions.mappers to recreate your custom object
    in the parse json string state just call this method with the json_string your complete object_type and with your
    mappers dict.
    the mappers dict must contain in the key the object_type (ex. User) and the value will contain a method that get
    key, value (the key will be the name of the object property we like to parse and the value
    will be the properties of the object)
    """
    obj = json.loads(json_string, cls=JsonDecoder, object_mapper=mappers.get(object_type, None))

    if obj is not None:
        try:
            obj = object_type(**obj)
        except TypeError:
            initialize_dict, set_needed = Utils.make_initialize_dict(obj, object_type.__init__)
            o = object_type(**initialize_dict)
            if set_needed:
                for key, value in obj.items():
                    setattr(o, key, value)
            obj = o
    return obj