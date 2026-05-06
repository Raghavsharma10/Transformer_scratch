def split_sig(params):
    """
    Split a list of parameters/types by commas,
    whilst respecting brackets.

    For example:
      String arg0, int arg2 = 1, List<int> arg3 = [1, 2, 3]
      => ['String arg0', 'int arg2 = 1', 'List<int> arg3 = [1, 2, 3]']
    """
    result = []
    current = ''
    level = 0
    for char in params:
        if char in ('<', '{', '['):
            level += 1
        elif char in ('>', '}', ']'):
            level -= 1
        if char != ',' or level > 0:
            current += char
        elif char == ',' and level == 0:
            result.append(current)
            current = ''
    if current.strip() != '':
        result.append(current)
    return result