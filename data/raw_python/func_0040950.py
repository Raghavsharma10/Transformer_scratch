def search_mergedcell_value(xl_sheet, merged_range):
    """
    Search for a value in merged_range cells.
    """
    for search_row_idx in range(merged_range[0], merged_range[1]):
        for search_col_idx in range(merged_range[2], merged_range[3]):
            if xl_sheet.cell(search_row_idx, search_col_idx).value:
                return xl_sheet.cell(search_row_idx, search_col_idx)
    return False