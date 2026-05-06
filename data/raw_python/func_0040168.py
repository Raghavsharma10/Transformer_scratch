def _format_response(rows, fields, unique_col_names):
    """This function will look at the data column of rows and extract the specified fields. It
    will also dedup changes where the specified fields have not changed. The list of rows should
    be ordered by the compound primary key which versioning pivots around and be in ascending
    version order.

    This function will return a list of dictionaries where each dictionary has the following
    schema:
        {
            'updated_at': timestamp of the change,
            'version': version number for the change,
            'data': a nested dictionary containing all keys specified in fields and values
                corresponding to values in the user table.
        }

    Note that some versions may be omitted in the output for the same key if the specified fields
    were not changed between versions.

    :param rows: a list of dictionaries representing rows from the ArchiveTable.
    :param fields: a list of strings of fields to be extracted from the archived row.
    """
    output = []
    old_id = None
    for row in rows:
        id_ = {k: row[k] for k in unique_col_names}
        formatted = {k: row[k] for k in row if k != 'data'}
        if id_ != old_id:  # new unique versioned row
            data = row['data']
            formatted['data'] = {k: data.get(k) for k in fields}
            output.append(formatted)
        else:
            data = row['data']
            pruned_data = {k: data.get(k) for k in fields}
            if (
                pruned_data != output[-1]['data'] or
                row['deleted'] != output[-1]['deleted']
            ):
                formatted['data'] = pruned_data
                output.append(formatted)
        old_id = id_
    return output