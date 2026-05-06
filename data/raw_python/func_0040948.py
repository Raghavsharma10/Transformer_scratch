def csv_tolist(path_to_file, **kwargs):
    """
    Parse the csv file to a list of rows.
    """

    result = []

    encoding = kwargs.get('encoding', 'utf-8')
    delimiter = kwargs.get('delimiter', ',')
    dialect = kwargs.get('dialect', csv.excel)

    _, _ext = path_to_file.split('.', 1)

    try:

        file = codecs.open(path_to_file, 'r', encoding)
        items_file = io.TextIOWrapper(file, encoding=encoding)
        result = list(
            csv.reader(items_file, delimiter=delimiter, dialect=dialect))

        items_file.close()
        file.close()

    except Exception as ex:
        result = []
        logger.error('Fail parsing csv to list of rows - {}'.format(ex))

    return result