def find_font(face, bold, italic):
    """Find font"""
    bold = FC_WEIGHT_BOLD if bold else FC_WEIGHT_REGULAR
    italic = FC_SLANT_ITALIC if italic else FC_SLANT_ROMAN
    face = face.encode('utf8')
    fontconfig.FcInit()
    pattern = fontconfig.FcPatternCreate()
    fontconfig.FcPatternAddInteger(pattern, FC_WEIGHT, bold)
    fontconfig.FcPatternAddInteger(pattern, FC_SLANT, italic)
    fontconfig.FcPatternAddString(pattern, FC_FAMILY, face)
    fontconfig.FcConfigSubstitute(0, pattern, FcMatchPattern)
    fontconfig.FcDefaultSubstitute(pattern)
    result = FcType()
    match = fontconfig.FcFontMatch(0, pattern, byref(result))
    fontconfig.FcPatternDestroy(pattern)
    if not match:
        raise RuntimeError('Could not match font "%s"' % face)
    value = FcValue()
    fontconfig.FcPatternGet(match, FC_FAMILY, 0, byref(value))
    if(value.u.s != face):
        warnings.warn('Could not find face match "%s", falling back to "%s"'
                      % (face, value.u.s))
    result = fontconfig.FcPatternGet(match, FC_FILE, 0, byref(value))
    if result != 0:
        raise RuntimeError('No filename or FT face for "%s"' % face)
    fname = value.u.s
    return fname.decode('utf-8')