def rst_to_json(text):
    """ I convert Restructured Text with field lists into Dictionaries!

        TODO: Convert to text node approach.
    """
    records = []
    last_type = None
    key = None
    data = {}
    directive = False

    lines = text.splitlines()
    for index, line in enumerate(lines):

        # check for directives
        if len(line) and line.strip().startswith(".."):
            directive = True
            continue

        # set the title
        if len(line) and (line[0] in string.ascii_letters or line[0].isdigit()):
            directive = False
            try:
                if lines[index + 1][0] not in DIVIDERS:
                    continue
            except IndexError:
                continue
            data = text_cleanup(data, key, last_type)
            data = {"title": line.strip()}
            records.append(
                data
            )
            continue

        # Grab standard fields (int, string, float)
        if len(line) and line[0].startswith(":"):
            data = text_cleanup(data, key, last_type)
            index = line.index(":", 1)
            key = line[1:index]
            value = line[index + 1:].strip()
            data[key], last_type = type_converter(value)
            directive = False
            continue

        # Work on multi-line strings
        if len(line) and line[0].startswith(" ") and directive == False:
            if not isinstance(data[key], str):
                # Not a string so continue on
                continue
            value = line.strip()
            if not len(value):
                # empty string, continue on
                continue
            # add next line
            data[key] += "\n{}".format(value)
            continue

        if last_type == STRING_TYPE and not len(line):
            if key in data.keys():
                data[key] += "\n"

    return json.dumps(records)