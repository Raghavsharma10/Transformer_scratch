def std_table(rows):
    """Return a formatted table of given rows"""

    result = ""
    if len(rows) > 1:
        headers = rows[0]._fields
        lens = []
        for i in range(len(rows[0])):
            lens.append(len(max([x[i] for x in rows] + [headers[i]],
                                key=lambda x: len(str(x)))))
        formats = []
        hformats = []
        for i in range(len(rows[0])):
            if isinstance(rows[0][i], int):
                formats.append("%%%dd" % lens[i])
            else:
                formats.append("%%-%ds" % lens[i])
            hformats.append("%%-%ds" % lens[i])
        pattern = " | ".join(formats)
        hpattern = " | ".join(hformats)
        separator = "-+-".join(['-' * n for n in lens])
        result += hpattern % tuple(headers) + " \n"
        result += separator + "\n"

        for line in rows:
            result += pattern % tuple(t for t in line) + "\n"
    elif len(rows) == 1:
        row = rows[0]
        hwidth = len(max(row._fields, key=lambda x: len(x)))
        for i in range(len(row)):
            result += "%*s = %s" % (hwidth, row._fields[i], row[i]) + "\n"

    return result