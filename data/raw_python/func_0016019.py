def vocab_hash_algo(instance):
    """Ensure objects with 'hashes' properties only use values from the
    hash-algo-ov vocabulary.
    """
    for key, obj in instance['objects'].items():
        if 'type' not in obj:
            continue

        if obj['type'] == 'file':
            try:
                hashes = obj['hashes']
            except KeyError:
                pass
            else:
                for h in hashes:
                    if not (valid_hash_value(h)):
                        yield JSONError("Object '%s' has a 'hashes' dictionary"
                                        " with a hash of type '%s', which is not a "
                                        "value in the hash-algo-ov vocabulary nor a "
                                        "custom value prepended with 'x_'."
                                        % (key, h), instance['id'], 'hash-algo')

            try:
                ads = obj['extensions']['ntfs-ext']['alternate_data_streams']
            except KeyError:
                pass
            else:
                for datastream in ads:
                    if 'hashes' not in datastream:
                        continue
                    for h in datastream['hashes']:
                        if not (valid_hash_value(h)):
                            yield JSONError("Object '%s' has an NTFS extension"
                                            " with an alternate data stream that has a"
                                            " 'hashes' dictionary with a hash of type "
                                            "'%s', which is not a value in the "
                                            "hash-algo-ov vocabulary nor a custom "
                                            "value prepended with 'x_'."
                                            % (key, h), instance['id'], 'hash-algo')

            try:
                head_hashes = obj['extensions']['windows-pebinary-ext']['file_header_hashes']
            except KeyError:
                pass
            else:
                for h in head_hashes:
                    if not (valid_hash_value(h)):
                        yield JSONError("Object '%s' has a Windows PE Binary "
                                        "File extension with a file header hash of "
                                        "'%s', which is not a value in the "
                                        "hash-algo-ov vocabulary nor a custom value "
                                        "prepended with 'x_'."
                                        % (key, h), instance['id'], 'hash-algo')

            try:
                hashes = obj['extensions']['windows-pebinary-ext']['optional_header']['hashes']
            except KeyError:
                pass
            else:
                for h in hashes:
                    if not (valid_hash_value(h)):
                        yield JSONError("Object '%s' has a Windows PE Binary "
                                        "File extension with an optional header that "
                                        "has a hash of '%s', which is not a value in "
                                        "the hash-algo-ov vocabulary nor a custom "
                                        "value prepended with 'x_'."
                                        % (key, h), instance['id'], 'hash-algo')

            try:
                sections = obj['extensions']['windows-pebinary-ext']['sections']
            except KeyError:
                pass
            else:
                for s in sections:
                    if 'hashes' not in s:
                        continue
                    for h in s['hashes']:
                        if not (valid_hash_value(h)):
                            yield JSONError("Object '%s' has a Windows PE "
                                            "Binary File extension with a section that"
                                            " has a hash of '%s', which is not a value"
                                            " in the hash-algo-ov vocabulary nor a "
                                            "custom value prepended with 'x_'."
                                            % (key, h), instance['id'], 'hash-algo')

        elif obj['type'] == 'artifact' or obj['type'] == 'x509-certificate':
            try:
                hashes = obj['hashes']
            except KeyError:
                pass
            else:
                for h in hashes:
                    if not (valid_hash_value(h)):
                        yield JSONError("Object '%s' has a 'hashes' dictionary"
                                        " with a hash of type '%s', which is not a "
                                        "value in the hash-algo-ov vocabulary nor a "
                                        "custom value prepended with 'x_'."
                                        % (key, h), instance['id'], 'hash-algo')