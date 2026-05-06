def translate_file_diff(data):
    """ Translates data where data["Type"]=="Diff" """
    diff = '\\FloatBarrier \section{Configuration}'
    sections = data.get('Data')
    for title, config in sections.items():
        title = title.replace('_', '\_')
        diff += ' \n \\subsection{$NAME}'.replace('$NAME', title)
        for opt, vals in config.items():
            opt = opt.replace('_', '\_')
            diff += '\n\n \\texttt{$NAME} : '.replace('$NAME', opt)
            if vals[0]:
                diff += '$NAME'.replace('$NAME', vals[-1])
            else:
                diff += ('{} \\textit{{{}}}'.format(vals[1], vals[-1]))
    diff += '\n\n'
    return diff