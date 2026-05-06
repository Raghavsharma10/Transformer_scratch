def translate_table(data):
    """ Translates data where data["Type"]=="Table" """
    headers = sorted(data.get("Headers", []))
    table = '\\FloatBarrier \n \\section{$NAME} \n'.replace('$NAME', data.get("Title", "table"))
    table += '\\begin{table}[!ht] \n \\begin{center}'

    # Set the number of columns
    n_cols = "c"*(len(headers)+1)
    table += '\n \\begin{tabular}{$NCOLS} \n'.replace("$NCOLS", n_cols)

    # Put in the headers
    for header in headers:
        table += ' $HEADER &'.replace('$HEADER', header).replace('%', '\%')
    table = table[:-1] + ' \\\\ \n \hline \n'

    # Put in the data
    for header in headers:
        table += ' $VAL &'.replace("$VAL", str(data["Data"][header]))
    table = table[:-1] + ' \\\\ \n \hline'
    table += '\n \end{tabular} \n \end{center} \n \end{table}\n'
    return table