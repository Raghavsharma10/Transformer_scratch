def padd(text, padding="top", size=1):
    """ Adds extra new lines to the top, bottom or both of a String

        @text: #str text to pad
        @padding: #str 'top', 'bottom' or 'all'
        @size: #int number of new lines

        -> #str padded @text
        ..
            from vital.debug import *

            padd("Hello world")
            # -> '\\nHello world'

            padd("Hello world", size=5, padding="all")
            # -> '\\n\\n\\n\\n\\nHello world\\n\\n\\n\\n\\n'
        ..
    """
    if padding:
        padding = padding.lower()
        pad_all = padding == 'all'
        padding_top = ""
        if padding and (padding == 'top' or pad_all):
            padding_top = "".join("\n" for x in range(size))
        padding_bottom = ""
        if padding and (padding == 'bottom' or pad_all):
            padding_bottom = "".join("\n" for x in range(size))
        return "{}{}{}".format(padding_top, text, padding_bottom)
    return text