def read_nem_file(file_path: str) -> NEMFile:
    """ Read in NEM file and return meter readings named tuple

    :param file_path: The NEM file to process
    :returns: The file that was created
    """

    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.zip':
        with zipfile.ZipFile(file_path, 'r') as archive:
            for csv_file in archive.namelist():
                with archive.open(csv_file) as csv_text:
                    # Zip file is open in binary mode
                    # So decode then convert back to list
                    nmi_file = csv_text.read().decode('utf-8').splitlines()
                    reader = csv.reader(nmi_file, delimiter=',')
                    return parse_nem_rows(reader, file_name=csv_file)

    with open(file_path) as nmi_file:
        return parse_nem_file(nmi_file)