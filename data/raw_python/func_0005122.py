def cut(text, length=50, replace_with="…"):
    """ Shortens text to @length, appends @replace_with to end of string
        if the string length is > @length

        @text: #str text to shortens
        @length: #int max length of string
        @replace_with: #str to replace chars beyond @length with
        ..
            from vital.debug import cut

            cut("Hello world", 8)
            # -> 'Hello w…'

            cut("Hello world", 15)
            # -> 'Hello world'
        ..
    """
    text_len = len(uncolorize(text))
    if text_len > length:
        replace_len = len(replace_with)
        color_spans = [
            _colors.span() for _colors in _find_colors.finditer(text)]
        chars = 0
        _length = length+1 - replace_len
        for i, c in enumerate(text):
            broken = False
            for span in color_spans:
                if span[0] <= i < span[1]:
                    broken = True
                    break
            if broken:
                continue
            chars += 1
            if chars <= _length:
                cutoff = i
            else:
                break
        if color_spans:
            return text[:cutoff] + replace_with + colors.RESET
        else:
            return text[:cutoff] + replace_with
    return text