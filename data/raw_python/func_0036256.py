def get_num_columns_and_rows(widths, gap_width, term_width):
    '''Given a list of string widths, a width of the minimum gap to place
    between them, and the maximum width of the output (such as a terminal
    width), calculate the number of columns and rows, and the width of each
    column, for the optimal layout.

    '''
    def calc_longest_width(widths, gap_width, ncols):
        longest = 0
        rows = [widths[s:s + ncols] for s in range(0, len(widths), ncols)]
        col_widths = rows[0] # Column widths start at the first row widths
        for r in rows:
            for ii, c in enumerate(r):
                if c > col_widths[ii]:
                    col_widths[ii] = c
            length = sum(col_widths) + gap_width * (ncols - 1)
            if length > longest:
                longest = length
        return longest, col_widths

    def calc_num_rows(num_items, cols):
        div, mod = divmod(num_items, cols)
        return div + (mod != 0)

    # Start with one row
    ncols = len(widths)
    # Calculate the width of the longest row as the longest set of item widths
    # ncols long and gap widths (gap_width * ncols - 1) that fits within the
    # terminal width.
    while ncols > 0:
        longest_width, col_widths = calc_longest_width(widths, gap_width, ncols)
        if longest_width < term_width:
            # This number of columns fits
            return calc_num_rows(len(widths), ncols), ncols, col_widths
        else:
            # This number of columns doesn't fit, so try one less
            ncols -= 1
    # If got here, it all has to go in one column
    return len(widths), 1, 0