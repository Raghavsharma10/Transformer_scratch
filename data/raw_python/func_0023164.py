def get_text_style(text):
    """Return the text style dict for a text instance"""
    style = {}
    style['alpha'] = text.get_alpha()
    if style['alpha'] is None:
        style['alpha'] = 1
    style['fontsize'] = text.get_size()
    style['color'] = color_to_hex(text.get_color())
    style['halign'] = text.get_horizontalalignment()  # left, center, right
    style['valign'] = text.get_verticalalignment()  # baseline, center, top
    style['malign'] = text._multialignment  # text alignment when '\n' in text
    style['rotation'] = text.get_rotation()
    style['zorder'] = text.get_zorder()
    return style