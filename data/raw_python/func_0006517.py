def prtcols(items, vpad=6):
    '''
    After computing the size of our rows and columns based on the terminal size
    and length of the largest element, use zip to aggregate our column lists
    into row lists and then iterate over the row lists and print them.
    '''
    from os import get_terminal_size
    items = list(items)  # copy list so we don't mutate it
    width, height = get_terminal_size()
    height -= vpad  # customize vertical padding
    pad = mkpad(items)
    rows = mkrows(items, pad, width, height)
    cols = mkcols(items, rows)
    # * operator in conjunction with zip, unzips the list
    for c in zip(*cols):
        row_format = '{:<{pad}}' * len(cols)
        print(row_format.format(*c, pad=pad))