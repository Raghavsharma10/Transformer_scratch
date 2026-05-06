def encode_collection(collection, encoding='utf-8'):
    """Encodes all the string keys and values in a collection with specified encoding"""

    if isinstance(collection, dict):
        return dict((encode_collection(key), encode_collection(value)) for key, value in collection.iteritems())
    elif isinstance(collection, list):
        return [encode_collection(element) for element in input]
    elif isinstance(collection, unicode):
        return collection.encode(encoding)
    else:
        return collection