def generate_lbryd_wrapper(url=LBRY_API_RAW_JSON_URL, read_file=__LBRYD_BASE_FPATH__, write_file=LBRYD_FPATH):
    """ Generates the actual functions for lbryd_api.py based on lbry's documentation

    :param str url: URL to the documentation we need to obtain,
     pybry.constants.LBRY_API_RAW_JSON_URL by default
    :param str read_file: This is the path to the file from which we will be reading
    :param str write_file: Path from project root to the file we'll be writing to.
     """

    functions = get_lbry_api_function_docs(url)

    # Open the actual file for appending
    with open(write_file, 'w') as lbry_file:

        lbry_file.write("# This file was generated at build time using the generator function\n")
        lbry_file.write("# You may edit but do so with caution\n")

        with open(read_file, 'r') as template:
            header = template.read()

        lbry_file.write(header)

        # Iterate through all the functions we retrieved
        for func in functions:

            method_definition = generate_method_definition(func)

            # Write to file
            lbry_file.write(method_definition)

    try:
        from yapf.yapflib.yapf_api import FormatFile

        # Now we should format the file using the yapf formatter
        FormatFile(write_file, in_place=True)

    except ImportError as IE:
        print("[Warning]: yapf is not installed, so the generated code will not follow an easy-to-read standard")
        print(IE)