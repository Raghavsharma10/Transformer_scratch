def _make_obj(obj):
    """Takes an object and returns a corresponding API class.

    The names and values of the data will match exactly with those found
    in the online docs at https://pokeapi.co/docsv2/ . In some cases, the data
    may be of a standard type, such as an integer or string. For those cases,
    the input value is simply returned, unchanged.

    :param obj: the object to be converted
    :return either the same value, if it does not need to be converted, or a
    APIResource or APIMetadata instance, depending on the data inputted.
    """

    if isinstance(obj, dict):
        if 'url' in obj.keys():
            url = obj['url']
            id_ = int(url.split('/')[-2])      # ID of the data.
            endpoint = url.split('/')[-3]  # Where the data is located.
            return APIResource(endpoint, id_, lazy_load=True)

        return APIMetadata(obj)

    return obj