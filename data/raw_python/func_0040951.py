def is_merged(sheet, row, column):
    """
    Check if a row, column cell is a merged cell
    """
    for cell_range in sheet.merged_cells:
        row_low, row_high, column_low, column_high = cell_range
        if (row in range(row_low, row_high)) and \
                (column in range(column_low, column_high)):

            # TODO: IS NECESARY THIS IF?
            if ((column_high - column_low) < sheet.ncols - 1) and \
                    ((row_high - row_low) < sheet.nrows - 1):
                return (True, cell_range)

    return False