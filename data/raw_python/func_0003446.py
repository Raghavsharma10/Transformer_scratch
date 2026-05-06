def emojificate_filter(content, autoescape=True):
    "Convert any emoji in a string into accessible content."
    # return mark_safe(emojificate(content))
    if autoescape:
        esc = conditional_escape
    else:
        esc = lambda x: x
    return mark_safe(emojificate(esc(content)))