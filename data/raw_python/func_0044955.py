def _delim_accum(control_files, filename_template, keys=None,
        exclude_keys=None, separator=DEFAULT_SEP, missing_action='fail'):
    """
    Accumulator for delimited files

    Combines each file with values from JSON dictionary in same directory

    :param iterable control_files: Iterable of control files
    :param filename_template: A template for the file to nest_map
    :param keys: List of keys to select from JSON dictionary. If ``None``, keep
                 all keys.
    :param separator: Delimiter
    """
    def map_fn(d, control, keys=keys):
        f = os.path.join(d, filename_template.format(**control))

        keys = keys if keys is not None else control.keys()
        if exclude_keys:
            keys = list(frozenset(keys) - frozenset(exclude_keys))
        if frozenset(keys) - frozenset(control):
            # Unknown keys
            raise ValueError(
                    "The following required key(s) are not present in {1}: {0}".format(
                        ', '.join(frozenset(keys) - frozenset(control)),
                        f))
        with open(f) as fp:
            reader = csv.DictReader(fp, delimiter=separator)
            for row in reader:
                row_dict = collections.OrderedDict(
                        itertools.chain(((k, row[k]) for k in reader.fieldnames),
                        ((k, v) for k, v in control.items() if k in keys)))

                yield row_dict
    if missing_action == 'warn':
        map_fn = _warn_on_io(map_fn)

    return itertools.chain.from_iterable(nest_map(control_files, map_fn))