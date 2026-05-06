def _load_glyph(f, char, glyphs_dict):
    """Load glyph from font into dict"""
    from ...ext.freetype import (FT_LOAD_RENDER, FT_LOAD_NO_HINTING,
                                 FT_LOAD_NO_AUTOHINT)
    flags = FT_LOAD_RENDER | FT_LOAD_NO_HINTING | FT_LOAD_NO_AUTOHINT
    face = _load_font(f['face'], f['bold'], f['italic'])
    face.set_char_size(f['size'] * 64)
    # get the character of interest
    face.load_char(char, flags)
    bitmap = face.glyph.bitmap
    width = face.glyph.bitmap.width
    height = face.glyph.bitmap.rows
    bitmap = np.array(bitmap.buffer)
    w0 = bitmap.size // height if bitmap.size > 0 else 0
    bitmap.shape = (height, w0)
    bitmap = bitmap[:, :width].astype(np.ubyte)

    left = face.glyph.bitmap_left
    top = face.glyph.bitmap_top
    advance = face.glyph.advance.x / 64.
    glyph = dict(char=char, offset=(left, top), bitmap=bitmap,
                 advance=advance, kerning={})
    glyphs_dict[char] = glyph
    # Generate kerning
    for other_char, other_glyph in glyphs_dict.items():
        kerning = face.get_kerning(other_char, char)
        glyph['kerning'][other_char] = kerning.x / 64.
        kerning = face.get_kerning(char, other_char)
        other_glyph['kerning'][char] = kerning.x / 64.