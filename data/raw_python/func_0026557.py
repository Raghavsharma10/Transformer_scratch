def section(rows, columns, items, label=None):
    """A section consisting of rows and columns"""

    # TODO: Integrate label

    sections = []

    column_class = "section-column col-sm-%i" % (12 / columns)

    for vertical in range(columns):
        column_items = []
        for horizontal in range(rows):
            try:
                item = items[horizontal][vertical]
                column_items.append(item)
            except IndexError:
                hfoslog('Field in', label, 'omitted, due to missing row/column:', vertical, horizontal,
                        lvl=warn, emitter='FORMS', tb=True, frame=2)

        column = {
            'type': 'section',
            'htmlClass': column_class,
            'items': column_items
        }
        sections.append(column)

    result = {
        'type': 'section',
        'htmlClass': 'row',
        'items': sections
    }

    return result