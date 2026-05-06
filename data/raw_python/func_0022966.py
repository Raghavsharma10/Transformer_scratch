def draw_texture(tex):
    """Draw a 2D texture to the current viewport

    Parameters
    ----------
    tex : instance of Texture2D
        The texture to draw.
    """
    from .program import Program
    program = Program(vert_draw, frag_draw)
    program['u_texture'] = tex
    program['a_position'] = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]
    program['a_texcoord'] = [[0., 1.], [0., 0.], [1., 1.], [1., 0.]]
    program.draw('triangle_strip')