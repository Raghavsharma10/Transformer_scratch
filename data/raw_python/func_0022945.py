def _get_k_p_a(font, left, right):
    """This actually calculates the kerning + advance"""
    # http://lists.apple.com/archives/coretext-dev/2010/Dec/msg00020.html
    # 1) set up a CTTypesetter
    chars = left + right
    args = [None, 1, cf.kCFTypeDictionaryKeyCallBacks,
            cf.kCFTypeDictionaryValueCallBacks]
    attributes = cf.CFDictionaryCreateMutable(*args)
    cf.CFDictionaryAddValue(attributes, kCTFontAttributeName, font)
    string = cf.CFAttributedStringCreate(None, CFSTR(chars), attributes)
    typesetter = ct.CTTypesetterCreateWithAttributedString(string)
    cf.CFRelease(string)
    cf.CFRelease(attributes)
    # 2) extract a CTLine from it
    range = CFRange(0, 1)
    line = ct.CTTypesetterCreateLine(typesetter, range)
    # 3) use CTLineGetOffsetForStringIndex to get the character positions
    offset = ct.CTLineGetOffsetForStringIndex(line, 1, None)
    cf.CFRelease(line)
    cf.CFRelease(typesetter)
    return offset