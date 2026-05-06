def List(column_names, title=None, boolstyle=None, editable=False, 
         select_col=None, sep='|', data=[], **kwargs):
    """Present a list of items to select.
    
    This will raise a Zenity List Dialog populated with the colomns and rows 
    specified and return either the cell or row that was selected or None if 
    the user hit cancel.
    
    column_names - A tuple or list containing the names of the columns.
    title - The title of the dialog box.
    boolstyle - Whether the first columns should be a bool option ("checklist",
                "radiolist") or None if it should be a text field.
    editable - True if the user can edit the cells.
    select_col - The column number of the selected cell to return or "ALL" to 
                 return the entire row.
    sep - Token to use as the row separator when parsing Zenity's return. 
          Cells should not contain this token.
    data - A list or tuple of tuples that contain the cells in the row.  The 
           size of the row's tuple must be equal to the number of columns.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = []
    for column in column_names:
        args.append('--column=%s' % column)
    
    if title:
        args.append('--title=%s' % title)
    if boolstyle:
        if not (boolstyle == 'checklist' or boolstyle == 'radiolist'):
            raise ValueError('"%s" is not a proper boolean column style.'
                             % boolstyle)
        args.append('--' + boolstyle)
    if editable:
        args.append('--editable')
    if select_col:
        args.append('--print-column=%s' % select_col)
    if sep != '|':
        args.append('--separator=%s' % sep)
    
    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    for datum in chain(*data):
        args.append(str(datum))
    
    p = run_zenity('--list', *args)

    if p.wait() == 0:
        return p.stdout.read().strip().split(sep)