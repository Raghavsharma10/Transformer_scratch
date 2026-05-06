def _format_level_2(rows, list_embeds, embed_many):
    """
    From the _format_level_1 function we have a list of rows. Because of using
    joins, we have as many rows as join result.

    For example:
    [{'id' : 'id1',
      'name' : 'name1,
      'b' : {'id': 'id2,
             'name': 'name2'}
     }
     {'id' : 'id1',
      'name' : 'name1,
      'b' : {'id' : 'id4',
             'name' : 'name4}
     }
    ]

    Here there is two elements which correspond to one rows because of the
    embed field 'b'. So we should transform it to:

    [{'id' : 'id1',
      'name' : 'name1,
      'b' : [{'id': 'id2,
             'name': 'name2'},
             {'id' : 'id4',
             'name' : 'name4}]
     }
    ]

    This is the purpose of this function.
    """

    def _uniqify_list(list_of_dicts):
        # list() for py34
        result = []
        set_ids = set()
        for v in list_of_dicts:
            if v['id'] in set_ids:
                continue
            set_ids.add(v['id'])
            result.append(v)
        return result

    row_ids_to_embed_values = {}
    for row in rows:
        # for each row, associate rows's id -> {all embeds values}
        if row['id'] not in row_ids_to_embed_values:
            row_ids_to_embed_values[row['id']] = {}
        # add embeds values to the current row
        for embd in list_embeds:
            if embd not in row:
                continue
            if embd not in row_ids_to_embed_values[row['id']]:
                # create a list or a dict depending on embed_many
                if embed_many[embd]:
                    row_ids_to_embed_values[row['id']][embd] = [row[embd]]
                else:
                    row_ids_to_embed_values[row['id']][embd] = row[embd]
            else:
                if embed_many[embd]:
                    row_ids_to_embed_values[row['id']][embd].append(row[embd])
        # uniqify each embed list
        for embd in list_embeds:
            if embd in row_ids_to_embed_values[row['id']]:
                embed_values = row_ids_to_embed_values[row['id']][embd]
                if isinstance(embed_values, list):
                    row_ids_to_embed_values[row['id']][embd] = _uniqify_list(embed_values)  # noqa
            else:
                row_ids_to_embed_values[row['id']][embd] = {}
                if embed_many[embd]:
                    row_ids_to_embed_values[row['id']][embd] = []

    # last loop over the initial rows in order to keep the ordering
    result = []
    # if row id in seen set then it means the row has been completely processed
    seen = set()
    for row in rows:
        if row['id'] in seen:
            continue
        seen.add(row['id'])
        new_row = {}
        # adds level 1 fields
        for field in row:
            if field not in list_embeds:
                new_row[field] = row[field]
        # adds all level 2 fields
        # list() for py34
        row_ids_to_embed_values_keys = list(row_ids_to_embed_values[new_row['id']].keys())  # noqa
        row_ids_to_embed_values_keys.sort()
        # adds the nested fields if there is somes
        for embd in list_embeds:
            if embd in row_ids_to_embed_values_keys:
                if '.' in embd:
                    prefix, suffix = embd.split('.', 1)
                    new_row[prefix][suffix] = row_ids_to_embed_values[new_row['id']][embd]  # noqa
                else:
                    new_row[embd] = row_ids_to_embed_values[new_row['id']][embd]  # noqa
            else:
                new_row_embd_value = {}
                if embed_many[embd]:
                    new_row_embd_value = []
                if '.' in embd:
                    prefix, suffix = embd.split('.', 1)
                    new_row[prefix][suffix] = new_row_embd_value
                else:
                    new_row[embd] = new_row_embd_value
        # row is complete !
        result.append(new_row)
    return result