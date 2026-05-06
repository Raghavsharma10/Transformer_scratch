def convert(csv, json, **kwargs):
    '''Convert csv to json.

    csv:  filename or file-like object
    json: filename  or file-like object


    if csv is '-' or None:
        stdin is used for input
    if json is '-' or None:
        stdout is used for output
    '''

    csv_local, json_local = None, None
    try:
        if csv == '-' or csv is None:
            csv = sys.stdin
        elif isinstance(csv, str):
            csv = csv_local = open(csv, 'r')

        if json == '-' or json is None:
            json = sys.stdout
        elif isinstance(json, str):
            json = json_local = open(json, 'w')

        data = load_csv(csv, **kwargs)
        save_json(data, json, **kwargs)
    finally:
        if csv_local is not None:
            csv_local.close()
        if json_local is not None:
            json_local.close()