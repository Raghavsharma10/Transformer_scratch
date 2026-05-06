def make_row_processors(bundle, source_headers, dest_table, env):
    """
    Make multiple row processors for all of the columns in a table.

    :param source_headers:

    :return:
    """

    dest_headers = [c.name for c in dest_table.columns]

    row_processors = []

    out = [file_header]

    transforms = list(dest_table.transforms)
    column_names = []
    column_types = []
    for i, segments in enumerate(transforms):

        seg_funcs = []

        for col_num, (segment, column) in enumerate(zip(segments, dest_table.columns), 1):

            if not segment:
                seg_funcs.append('row[{}]'.format(col_num - 1))
                continue

            assert column
            assert column.name == segment['column'].name
            col_name = column.name
            preamble, try_lines, exception = make_stack(env, i, segment)

            assert col_num == column.sequence_id, (dest_table.name, col_num, column.sequence_id)

            column_names.append(col_name)
            column_types.append(column.datatype)

            f_name = "{table_name}_{column_name}_{stage}".format(
                table_name=dest_table.name,
                column_name=col_name,
                stage=i
            )

            exception = (exception if exception else
                                  ('raise ValueError("Failed to cast column \'{}\', in '
                                   'function {}, value \'{}\': {}".format(header_d,"') + f_name +
                                  '", v.encode(\'ascii\', \'replace\'), exc) ) ')

            try:
                i_s = source_headers.index(column.name)
                header_s = column.name
                v = 'row[{}]'.format(i_s)

            except ValueError as e:

                i_s = 'None'
                header_s = None
                v = 'None' if col_num > 1 else 'row_n' # Give the id column the row number


            i_d = column.sequence_id - 1

            header_d = column.name

            template_args = dict(
                f_name=f_name,
                table_name=dest_table.name,
                column_name=col_name,
                stage=i,
                i_s=i_s,
                i_d=i_d,
                header_s=header_s,
                header_d=header_d,
                v=v,
                exception=indent + exception,
                stack='\n'.join(indent + l for l in try_lines),
                col_args = '# col_args not implemented yet'
            )

            seg_funcs.append(f_name
                             + ('({v}, {i_s}, {i_d}, {header_s}, \'{header_d}\', '
                                'row, row_n, errors, scratch, accumulator, pipe, bundle, source)')
                             .format(v=v, i_s=i_s, i_d=i_d, header_s="'" + header_s + "'" if header_s else 'None',
                                     header_d=header_d))

            out.append('\n'.join(preamble))

            out.append(column_template.format(**template_args))

        source_headers = dest_headers

        stack = '\n'.join("{}{}, # {}".format(indent,l,cn)
                          for l,cn, dt in zip(seg_funcs, column_names, column_types))

        out.append(row_template.format(
            table=dest_table.name,
            stage=i,
            stack=stack
        ))

        row_processors.append('row_{table}_{stage}'.format(stage=i, table=dest_table.name))

    # Add the final datatype cast, which is done seperately to avoid an unecessary function call.

    stack = '\n'.join("{}cast_{}(row[{}], '{}', errors),".format(indent, c.datatype, i, c.name)
                      for i, c in enumerate(dest_table.columns) )

    out.append(row_template.format(
        table=dest_table.name,
        stage=len(transforms),
        stack=stack
    ))

    row_processors.append('row_{table}_{stage}'.format(stage=len(transforms), table=dest_table.name))

    out.append('row_processors = [{}]'.format(','.join(row_processors)))

    return '\n'.join(out)