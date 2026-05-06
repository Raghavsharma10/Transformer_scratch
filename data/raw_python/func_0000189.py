def create_for(line, search_result):
    '''Create a new "for loop" line as a replacement for the original code.
    '''
    try:
        return line.format(search_result.group("indented_for"),
                           search_result.group("var"),
                           search_result.group("start"),
                           search_result.group("stop"),
                           search_result.group("cond"))
    except IndexError:
        return line.format(search_result.group("indented_for"),
                           search_result.group("var"),
                           search_result.group("start"),
                           search_result.group("stop"))