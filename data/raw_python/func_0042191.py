def build_graph(layout_str, slanted):
    '''
    builds an adjacency graph as a dictionary: {character: [adjacent_characters]}.
    adjacent characters occur in a clockwise order.
    for example:
    * on qwerty layout, 'g' maps to ['fF', 'tT', 'yY', 'hH', 'bB', 'vV']
    * on keypad layout, '7' maps to [None, None, None, '=', '8', '5', '4', None]
    '''
    position_table = {} # maps from tuple (x,y) -> characters at that position.
    tokens = layout_str.split()
    token_size = len(tokens[0])
    x_unit = token_size + 1 # x position unit length is token length plus 1 for the following whitespace.
    adjacency_func = get_slanted_adjacent_coords if slanted else get_aligned_adjacent_coords
    assert all(len(token) == token_size for token in tokens), 'token length mismatch:\n ' + layout_str
    for y, line in enumerate(layout_str.split('\n')):
        slant = y - 1 if slanted else 0 # the way i illustrated keys above, each qwerty row is indented one space in from the last
        for token in line.split():
            x, remainder = divmod(line.index(token) - slant, x_unit)
            assert remainder == 0, 'unexpected x offset for %s in:\n%s' % (token, layout_str)
            position_table[(x,y)] = token

    adjacency_graph = {}
    for (x,y), chars in position_table.items():
        for char in chars:
            adjacency_graph[char] = []
            for coord in adjacency_func(x, y):
                # position in the list indicates direction (for qwerty, 0 is left, 1 is top, 2 is top right, ...)
                # for edge chars like 1 or m, insert None as a placeholder when needed so that each character in the graph has a same-length adjacency list.
                adjacency_graph[char].append(position_table.get(coord, None))
    return adjacency_graph