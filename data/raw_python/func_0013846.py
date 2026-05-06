def append_csv_data(file_strings):
    """ Append data from multiple csv files for the same time period

    Parameters
    -----------
    file_strings : array-like
        Lists or arrays of strings, where each string contains one file of data

    Returns
    -------
    out_string : string
        String with all data, ready for output to a file
        
    """
    # Start with data from the first list element
    out_lines = list()
    head_line = None

    # Cycle through the lists of file strings, creating a list of line strings
    for fstrings in file_strings:
        file_lines = fstrings.split('\n')

        # Remove and save the header line
        head_line = file_lines.pop(0)

        # Save the data lines
        out_lines.extend(file_lines)

    # Sort the output lines by date and station (first two columns) in place
    out_lines.sort()

    # Remove all zero-length lines from front, add one to back, and add header
    i = 0
    while len(out_lines[i]) == 0:
        out_lines.pop(i)

    out_lines.insert(0, head_line)
    out_lines.append('')

    # Join the output lines into a single string
    out_string = "\n".join(out_lines)

    return out_string