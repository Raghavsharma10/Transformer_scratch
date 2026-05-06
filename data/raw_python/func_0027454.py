def _format_level_1(rows, root_table_name):
    """
    Transform sqlalchemy source:
    [{'a_id' : 'id1',
      'a_name' : 'name1,
      'b_id' : 'id2',
      'b_name' : 'name2},
     {'a_id' : 'id3',
      'a_name' : 'name3,
      'b_id' : 'id4',
      'b_name' : 'name4}
    ]
    to
    [{'id' : 'id1',
      'name': 'name2',
      'b' : {'id': 'id2', 'name': 'name2'},
     {'id' : 'id3',
      'name': 'name3',
      'b' : {'id': 'id4', 'name': 'name4'}
    ]
    """
    result_rows = []
    for row in rows:
        row = dict(row)
        result_row = {}
        prefixes_to_remove = []
        for field in row:
            if field.startswith('next_topic'):
                prefix = 'next_topic'
                suffix = field[11:]
                if suffix == 'id_1':
                    suffix = 'id'
            else:
                prefix, suffix = field.split('_', 1)
            if suffix == 'id' and row[field] is None:
                prefixes_to_remove.append(prefix)
            if prefix not in result_row:
                result_row[prefix] = {suffix: row[field]}
            else:
                result_row[prefix].update({suffix: row[field]})
        # remove field with id == null
        for prefix_to_remove in prefixes_to_remove:
            result_row.pop(prefix_to_remove)
        root_table_fields = result_row.pop(root_table_name)
        result_row.update(root_table_fields)
        result_rows.append(result_row)
    return result_rows