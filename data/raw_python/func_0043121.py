def split_qs(string, delimiter='&'):
    """Split a string by the specified unquoted, not enclosed delimiter"""

    open_list = '[<{('
    close_list = ']>})'
    quote_chars = '"\''

    level = index = last_index = 0
    quoted = False
    result = []

    for index, letter in enumerate(string):
        if letter in quote_chars:
            if not quoted:
                quoted = True
                level += 1
            else:
                quoted = False
                level -= 1
        elif letter in open_list:
            level += 1
        elif letter in close_list:
                level -= 1
        elif letter == delimiter and level == 0:
            # Split here
            element = string[last_index: index]
            if element:
                result.append(element)
            last_index = index + 1

    if index:
        element = string[last_index: index + 1]
        if element:
            result.append(element)

    return result