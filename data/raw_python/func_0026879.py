def parse_nem_file(nem_file) -> NEMFile:
    """ Parse NEM file and return meter readings named tuple """
    reader = csv.reader(nem_file, delimiter=',')
    return parse_nem_rows(reader, file_name=nem_file)