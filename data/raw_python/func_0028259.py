def adjust_column_width(worksheet):
    """Adjust column width in worksheet.

    Args:
        worksheet: worksheet to be adjusted
    """
    dims = {}
    padding = 1
    for row in worksheet.rows:
        for cell in row:
            if not cell.value:
                continue
            dims[cell.column] = max(
                dims.get(cell.column, 0),
                len(str(cell.value))
            )
    for col, value in list(dims.items()):
        worksheet.column_dimensions[col].width = value + padding