def hash_length(instance):
    """Ensure keys in 'hashes'-type properties are no more than 30 characters long.
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
                    if (len(h) > 30):
                        yield JSONError("Object '%s' has a 'hashes' dictionary"
                                        " with a hash of type '%s', which is "
                                        "longer than 30 characters."
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
                        if (len(h) > 30):
                            yield JSONError("Object '%s' has an NTFS extension"
                                            " with an alternate data stream that has a"
                                            " 'hashes' dictionary with a hash of type "
                                            "'%s', which is longer than 30 "
                                            "characters."
                                            % (key, h), instance['id'], 'hash-algo')

            try:
                head_hashes = obj['extensions']['windows-pebinary-ext']['file_header_hashes']
            except KeyError:
                pass
            else:
                for h in head_hashes:
                    if (len(h) > 30):
                        yield JSONError("Object '%s' has a Windows PE Binary "
                                        "File extension with a file header hash of "
                                        "'%s', which is longer than 30 "
                                        "characters."
                                        % (key, h), instance['id'], 'hash-algo')

            try:
                hashes = obj['extensions']['windows-pebinary-ext']['optional_header']['hashes']
            except KeyError:
                pass
            else:
                for h in hashes:
                    if (len(h) > 30):
                        yield JSONError("Object '%s' has a Windows PE Binary "
                                        "File extension with an optional header that "
                                        "has a hash of '%s', which is longer "
                                        "than 30 characters."
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
                        if (len(h) > 30):
                            yield JSONError("Object '%s' has a Windows PE "
                                            "Binary File extension with a section that"
                                            " has a hash of '%s', which is "
                                            "longer than 30 characters."
                                            % (key, h), instance['id'], 'hash-algo')

        elif obj['type'] == 'artifact' or obj['type'] == 'x509-certificate':
            try:
                hashes = obj['hashes']
            except KeyError:
                pass
            else:
                for h in hashes:
                    if (len(h) > 30):
                        yield JSONError("Object '%s' has a 'hashes' dictionary"
                                        " with a hash of type '%s', which is "
                                        "longer than 30 characters."
                                        % (key, h), instance['id'], 'hash-algo')