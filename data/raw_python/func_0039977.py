def parse_spss_headerfile(path, **kwargs):
    """
    Parse spss header file

    Arguments:
        path {str} -- path al fichero de cabecera.
        leyend_position -- posicion del la leyenda en el header.
    """
    headers_clean = {}
    try:
        with codecs.open(path, 'r', kwargs.get('encoding', 'latin-1')) as file_:
            raw_file = file_.read()
            raw_splited = exclude_empty_values(raw_file.split('.\r\n'))

            # suposse that by default spss leyend is in position 0.
            leyend = parse_spss_header_leyend(
                raw_leyend=raw_splited.pop(kwargs.get('leyend_position', 0)),
                leyend=headers_clean)

            if not leyend:
                raise Exception('Empty leyend')

            # only want VARIABLE(S) LABEL(S) & VALUE(S) LABEL(S)
            for label in [x for x in raw_splited if 'label' in x.lower()]:
                values = parse_spss_header_labels(
                    raw_labels=label,
                    headers=leyend)

    except Exception as ex:
        logger.error('Fail to parse spss headerfile - {}'.format(ex), header_file=path)
        headers_clean = {}

    return headers_clean