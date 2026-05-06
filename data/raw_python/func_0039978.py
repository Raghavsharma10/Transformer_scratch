def parse_spss_datafile(path, **kwargs):
    """
    Parse spss data file

    Arguments:
        path {str} -- path al fichero de cabecera.
        **kwargs {[dict]} -- otros argumentos que puedan llegar
    """
    data_clean = []
    with codecs.open(path, 'r', kwargs.get('encoding', 'latin-1')) as file_:
        raw_file = file_.read()
        data_clean = raw_file.split('\r\n')
    return exclude_empty_values(data_clean)