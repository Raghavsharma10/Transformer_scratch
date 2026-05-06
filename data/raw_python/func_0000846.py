def load_metadata(stream):
    """Load JSON metadata from opened stream."""
    try:
        metadata = json.load(
            stream, encoding='utf8', object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as e:
        err = RuntimeError('Error parsing {}: {}'.format(stream.name, e))
        raise_from(err, e)
    else:
        # convert changelog keys back to ints for sorting
        for group in metadata:
            if group == '$version':
                continue
            apis = metadata[group]['apis']
            for api in apis.values():
                int_changelog = OrderedDict()
                for version, log in api.get('changelog', {}).items():
                    int_changelog[int(version)] = log
                api['changelog'] = int_changelog
    finally:
        stream.close()

    return metadata