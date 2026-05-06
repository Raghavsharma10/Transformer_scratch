def translate_page(data):
    """
    Translates data elements with data['Type'] = 'page'.  This is the
    top level of translation that occurs, and delegates the translation
    of other element types contained on a page to their proper functions.
    """
    if "Page" != data["Type"]:
        return ""

    tex_str = ('\\documentclass{article}\\n' +
               '\\usepackage{placeins}\\n' +
               '\\title{LIVVkit}\\n' +
               '\\author{$USER}\\n' +
               '\\usepackage[parfill]{parskip}\\n' +
               '\\begin{document}\\n' +
               '\\maketitle\\n'
               ).replace('$USER', livvkit.user)

    content = data["Data"]
    for tag_name in ["Elements", "Tabs"]:
        for tag in content.get(tag_name, []):
                print("Translating " + tag["Type"])
                tex_str += translate_map[tag["Type"]](tag)

    tex_str += '\n\\end{document}'
    return tex_str