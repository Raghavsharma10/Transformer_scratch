def translate_summary(data):
    """ Translates data where data["Type"]=="Summary" """
    headers = sorted(data.get("Headers", []))
    summary = '\\FloatBarrier \n \\section{$NAME} \n'.replace('$NAME', data.get("Title", "table"))
    summary += ' \\begin{table}[!ht] \n \\begin{center}'

    # Set the number of columns
    n_cols = len(headers)
    col_str = "l" + "c" * n_cols
    summary += '\n \\begin{tabular}{$NCOLS} \n'.replace("$NCOLS", col_str)
    spacer = ' &' * n_cols + r'\\[.5em]'

    for header in headers:
        summary += '& $HEADER '.replace('$HEADER', header).replace('%', '\%')
    summary += ' \\\\ \hline \n'

    names = sorted(six.iterkeys(data.get("Data", [])))
    for name in names:
        summary += '\n\n \\textbf{{{}}} {} \n'.format(name, spacer)
        cases = data.get("Data", []).get(name, {})
        for case, c_data in cases.items():
            summary += ' $CASE & '.replace('$CASE', str(case))
            for header in headers:
                h_data = c_data.get(header, "")
                if list is type(h_data) and len(h_data) == 2:
                    summary += (' $H_DATA_0 of $H_DATA_1 &'
                                .replace('$H_DATA_0', str(h_data[0]))
                                .replace('$H_DATA_1', str(h_data[1]))
                                .replace('%', '\%'))
                else:
                    summary += ' $H_DATA &'.replace('$H_DATA', str(h_data)).replace('%', '\%')

            # This takes care of the trailing & that comes from processing the headers.
            summary = summary[:-1] + r' \\'

    summary += '\n \end{tabular} \n \end{center} \n \end{table}\n'
    return summary