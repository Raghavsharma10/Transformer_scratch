def read_excel(filename, sheet=None):
    """Reads an Excel file containing a tabular description of a transition function,
       as found in Sipser. Major difference: instead of multiple header rows,
       only a single header row whose entries might be tuples.
       """

    from openpyxl import load_workbook
    wb = load_workbook(filename)
    if sheet is None:
        ws = wb.active
    else:
        ws = wb.get_sheet_by_name(sheet)
    table = [[cell.value or "" for cell in row] for row in ws.rows]
    return from_table(table)