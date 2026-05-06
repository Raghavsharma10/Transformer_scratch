def _text_to_vbo(text, font, anchor_x, anchor_y, lowres_size):
    """Convert text characters to VBO"""
    # Necessary to flush commands before requesting current viewport because
    # There may be a set_viewport command waiting in the queue.
    # TODO: would be nicer if each canvas just remembers and manages its own
    # viewport, rather than relying on the context for this.
    canvas = context.get_current_canvas()
    canvas.context.flush_commands()

    text_vtype = np.dtype([('a_position', np.float32, 2),
                           ('a_texcoord', np.float32, 2)])
    vertices = np.zeros(len(text) * 4, dtype=text_vtype)
    prev = None
    width = height = ascender = descender = 0
    ratio, slop = 1. / font.ratio, font.slop
    x_off = -slop
    # Need to make sure we have a unicode string here (Py2.7 mis-interprets
    # characters like "•" otherwise)
    if sys.version[0] == '2' and isinstance(text, str):
        text = text.decode('utf-8')
    # Need to store the original viewport, because the font[char] will
    # trigger SDF rendering, which changes our viewport
    # todo: get rid of call to glGetParameter!
    orig_viewport = canvas.context.get_viewport()
    for ii, char in enumerate(text):
        glyph = font[char]
        kerning = glyph['kerning'].get(prev, 0.) * ratio
        x0 = x_off + glyph['offset'][0] * ratio + kerning
        y0 = glyph['offset'][1] * ratio + slop
        x1 = x0 + glyph['size'][0]
        y1 = y0 - glyph['size'][1]
        u0, v0, u1, v1 = glyph['texcoords']
        position = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
        texcoords = [[u0, v0], [u0, v1], [u1, v1], [u1, v0]]
        vi = ii * 4
        vertices['a_position'][vi:vi+4] = position
        vertices['a_texcoord'][vi:vi+4] = texcoords
        x_move = glyph['advance'] * ratio + kerning
        x_off += x_move
        ascender = max(ascender, y0 - slop)
        descender = min(descender, y1 + slop)
        width += x_move
        height = max(height, glyph['size'][1] - 2*slop)
        prev = char
    # Also analyse chars with large ascender and descender, otherwise the
    # vertical alignment can be very inconsistent
    for char in 'hy':
        glyph = font[char]
        y0 = glyph['offset'][1] * ratio + slop
        y1 = y0 - glyph['size'][1]
        ascender = max(ascender, y0 - slop)
        descender = min(descender, y1 + slop)
        height = max(height, glyph['size'][1] - 2*slop)

    if orig_viewport is not None:
        canvas.context.set_viewport(*orig_viewport)

    # Tight bounding box (loose would be width, font.height /.asc / .desc)
    width -= glyph['advance'] * ratio - (glyph['size'][0] - 2*slop)
    dx = dy = 0
    if anchor_y == 'top':
        dy = -ascender
    elif anchor_y in ('center', 'middle'):
        dy = -(height / 2 + descender)
    elif anchor_y == 'bottom':
        dy = -descender
    # Already referenced to baseline
    # elif anchor_y == 'baseline':
    #     dy = -descender
    if anchor_x == 'right':
        dx = -width
    elif anchor_x == 'center':
        dx = -width / 2.
    vertices['a_position'] += (dx, dy)
    vertices['a_position'] /= lowres_size
    return vertices