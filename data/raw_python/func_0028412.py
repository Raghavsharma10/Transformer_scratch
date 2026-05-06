def write_data(worksheet, data):
    """Writes data into worksheet.

    Args:
        worksheet: worksheet to write into
        data: data to be written
    """
    if not data:
        return

    if isinstance(data, list):
        rows = data
    else:
        rows = [data]

    if isinstance(rows[0], dict):
        keys = get_keys(rows)
        worksheet.append([utilities.convert_snake_to_title_case(key) for key in keys])
        for row in rows:
            values = [get_value_from_row(row, key) for key in keys]
            worksheet.append(values)
    elif isinstance(rows[0], list):
        for row in rows:
            values = [utilities.normalize_cell_value(value) for value in row]
            worksheet.append(values)
    else:
        for row in rows:
            worksheet.append([utilities.normalize_cell_value(row)])