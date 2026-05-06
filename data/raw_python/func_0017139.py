def format_table(table, column_names=None, column_specs=None, max_col_width=32, auto_col_width=False):
    """
    Table pretty printer. Expects tables to be given as arrays of arrays::

        print(format_table([[1, "2"], [3, "456"]], column_names=['A', 'B']))
    """
    orig_col_args = dict(column_names=column_names, column_specs=column_specs)
    if len(table) > 0:
        col_widths = [0] * len(table[0])
    elif column_specs is not None:
        col_widths = [0] * (len(column_specs) + 1)
    elif column_names is not None:
        col_widths = [0] * len(column_names)
    my_col_names, id_column = [], None
    if column_specs is not None:
        column_names = ["Row"]
        column_names.extend([col["name"] for col in column_specs])
        column_specs = [{"name": "Row", "type": "float"}] + column_specs
    if column_names is not None:
        for i in range(len(column_names)):
            if column_names[i].lower() == "id":
                id_column = i
            my_col = ansi_truncate(str(column_names[i]), max_col_width if i not in {0, id_column} else 99)
            my_col_names.append(my_col)
            col_widths[i] = max(col_widths[i], len(strip_ansi_codes(my_col)))
    trunc_table = []
    for row in table:
        my_row = []
        for i in range(len(row)):
            my_item = ansi_truncate(str(row[i]), max_col_width if i not in {0, id_column} else 99)
            my_row.append(my_item)
            col_widths[i] = max(col_widths[i], len(strip_ansi_codes(my_item)))
        trunc_table.append(my_row)

    type_colormap = {"boolean": BLUE(),
                     "integer": YELLOW(),
                     "float": WHITE(),
                     "string": GREEN()}
    for i in "uint8", "int16", "uint16", "int32", "uint32", "int64":
        type_colormap[i] = type_colormap["integer"]
    type_colormap["double"] = type_colormap["float"]

    def col_head(i):
        if column_specs is not None:
            return BOLD() + type_colormap[column_specs[i]["type"]] + column_names[i] + ENDC()
        else:
            return BOLD() + WHITE() + column_names[i] + ENDC()

    formatted_table = [border("┌") + border("┬").join(border("─") * i for i in col_widths) + border("┐")]
    if len(my_col_names) > 0:
        padded_column_names = [col_head(i) + " " * (col_widths[i] - len(my_col_names[i]))
                               for i in range(len(my_col_names))]
        formatted_table.append(border("│") + border("│").join(padded_column_names) + border("│"))
        formatted_table.append(border("├") + border("┼").join(border("─") * i for i in col_widths) + border("┤"))

    for row in trunc_table:
        padded_row = [row[i] + " " * (col_widths[i] - len(strip_ansi_codes(row[i]))) for i in range(len(row))]
        formatted_table.append(border("│") + border("│").join(padded_row) + border("│"))
    formatted_table.append(border("└") + border("┴").join(border("─") * i for i in col_widths) + border("┘"))

    if auto_col_width:
        if not sys.stdout.isatty():
            raise AegeaException("Cannot auto-format table, output is not a terminal")
        table_width = len(strip_ansi_codes(formatted_table[0]))
        tty_cols, tty_rows = get_terminal_size()
        if table_width > max(tty_cols, 80):
            return format_table(table, max_col_width=max_col_width - 1, auto_col_width=True, **orig_col_args)
    return "\n".join(formatted_table)