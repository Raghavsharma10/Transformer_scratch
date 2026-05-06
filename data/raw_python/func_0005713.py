def _generate_csv_header_line(*, header_names, header_prefix='', header=True, sep=',', newline='\n'):
    """
    Helper function to generate a CSV header line depending on
    the combination of arguments provided.
    """
    if isinstance(header, str):  # user-provided header line
        header_line = header + newline
    else:
        if not (header is None or isinstance(header, bool)):
            raise ValueError(f"Invalid value for argument `header`: {header}")
        else:
            if header:
                header_line = header_prefix + sep.join(header_names) + newline
            else:
                header_line = ""
    return header_line