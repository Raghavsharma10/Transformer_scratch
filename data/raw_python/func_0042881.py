def table2csv(table_data):
    """
    :param table_data: expecting a list of tuples
    :return: text in nice formatted csv
    """
    text_data = [tuple(value2json(vals, pretty=True) for vals in rows) for rows in table_data]

    col_widths = [max(len(text) for text in cols) for cols in zip(*text_data)]
    template = ", ".join(
        "{{" + text_type(i) + "|left_align(" + text_type(w) + ")}}"
        for i, w in enumerate(col_widths)
    )
    text = "\n".join(expand_template(template, d) for d in text_data)
    return text