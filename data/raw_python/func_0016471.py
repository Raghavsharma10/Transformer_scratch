def register_json():
    """Register a encoder/decoder for JSON serialization."""
    from anyjson import serialize as json_serialize
    from anyjson import deserialize as json_deserialize

    registry.register('json', json_serialize, json_deserialize,
                      content_type='application/json',
                      content_encoding='utf-8')