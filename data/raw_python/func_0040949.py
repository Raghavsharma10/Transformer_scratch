def excel_todictlist(path_to_file, **kwargs):
    """
    Parse excel file to a dict list of sheets, rows.
    """
    result = collections.OrderedDict()
    encoding = kwargs.get('encoding', 'utf-8')
    formatting_info = '.xlsx' not in path_to_file
    count = 0

    with xlrd.open_workbook(
        path_to_file,
        encoding_override=encoding, formatting_info=formatting_info) \
            as _excelfile:

        for sheet_name_raw in _excelfile.sheet_names():

            # if empty sheet name put sheet# as name
            sheet_name = sheet_name_raw or "sheet{}".format(count)
            result[sheet_name] = []

            xl_sheet = _excelfile.sheet_by_name(sheet_name_raw)

            for row_idx in range(0, xl_sheet.nrows):
                col_data = []
                for col_idx in range(0, xl_sheet.ncols):

                    # Get cell object by row, col
                    cell_obj = xl_sheet.cell(row_idx, col_idx)
                    merged_info = is_merged(xl_sheet, row_idx, col_idx)

                    # Search for value in merged_info
                    if not cell_obj.value and merged_info:
                        cell_obj = search_mergedcell_value(
                            xl_sheet, merged_info[1])
                        col_data.append(cell_obj.value if cell_obj else '')
                    else:
                        col_data.append(cell_obj.value)

                result[sheet_name].append(col_data)

            count += 1  # increase sheet counter

    return result