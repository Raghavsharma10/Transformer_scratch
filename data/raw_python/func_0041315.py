def select_field(col, field_or_fields, filters=None):
    """Select single or multiple fields.

    :params field_or_fields: str or list of str
    :returns headers: headers
    :return data: list of row

    **中文文档**

    - 在选择单列时, 返回的是 str, list.
    - 在选择多列时, 返回的是 str list, list of list.

    返回单列或多列的数据。
    """
    fields = _preprocess_field_or_fields(field_or_fields)

    if filters is None:
        filters = dict()

    wanted = {field: True for field in fields}

    if len(fields) == 1:
        header = fields[0]
        data = [doc.get(header) for doc in col.find(filters, wanted)]
        return header, data
    else:
        headers = list(fields)
        data = [[doc.get(header) for header in headers]
                for doc in col.find(filters, wanted)]
        return headers, data