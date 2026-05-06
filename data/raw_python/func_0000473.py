def line_protocol(name, tags: dict = None, fields: dict = None, timestamp: float = None) -> str:
    """
    Format a report as per InfluxDB line protocol

    :param name: name of the report
    :param tags: tags identifying the specific report
    :param fields: measurements of the report
    :param timestamp: when the measurement was taken, in **seconds** since the epoch
    """
    output_str = name
    if tags:
        output_str += ','
        output_str += ','.join('%s=%s' % (key, value) for key, value in sorted(tags.items()))
    output_str += ' '
    output_str += ','.join(('%s=%r' % (key, value)).replace("'", '"') for key, value in sorted(fields.items()))
    if timestamp is not None:
        # line protocol requires nanosecond precision, python uses seconds
        output_str += ' %d' % (timestamp * 1E9)
    return output_str + '\n'