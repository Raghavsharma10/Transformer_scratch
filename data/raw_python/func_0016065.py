def vocab_encryption_algo(instance):
    """Ensure file objects' 'encryption_algorithm' property is from the
    encryption-algo-ov vocabulary.
    """
    for key, obj in instance['objects'].items():
        if 'type' in obj and obj['type'] == 'file':
            try:
                enc_algo = obj['encryption_algorithm']
            except KeyError:
                continue
            if enc_algo not in enums.ENCRYPTION_ALGO_OV:
                yield JSONError("Object '%s' has an 'encryption_algorithm' of "
                                "'%s', which is not a value in the "
                                "encryption-algo-ov vocabulary."
                                % (key, enc_algo), instance['id'],
                                'encryption-algo')