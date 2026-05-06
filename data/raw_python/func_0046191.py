def make_index(css_class, entities):
    """
    Generate the HTML index (a short description and a link to the full
    documentation) for a list of FunctionDocs or ClassDocs.
    """
    def make_entry(entity):
        return ('<dt><a href = "%(url)s">%(name)s</a></dt>\n' +
                '<dd>%(doc)s</dd>') % {
            'name': entity.name,
            'url': entity.url,
            'doc': first_sentence(entity.doc)
        }
    entry_text = '\n'.join(make_entry(val) for val in entities)
    if entry_text:
        return '<dl class = "%s">\n%s\n</dl>' % (css_class, entry_text)
    else:
        return ''