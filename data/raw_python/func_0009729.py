def process_tables(key, value, fmt, meta):
    """Processes the attributed tables."""

    global has_unnumbered_tables  # pylint: disable=global-statement

    # Process block-level Table elements
    if key == 'Table':

        # Inspect the table
        if len(value) == 5:  # Unattributed, bail out
            has_unnumbered_tables = True
            if fmt in ['latex']:
                return [RawBlock('tex', r'\begin{no-prefix-table-caption}'),
                        Table(*value),
                        RawBlock('tex', r'\end{no-prefix-table-caption}')]
            return None

        # Process the table
        table = _process_table(value, fmt)

        # Context-dependent output
        attrs = table['attrs']
        if table['is_unnumbered']:
            if fmt in ['latex']:
                return [RawBlock('tex', r'\begin{no-prefix-table-caption}'),
                        AttrTable(*value),
                        RawBlock('tex', r'\end{no-prefix-table-caption}')]

        elif fmt in ['latex']:
            if table['is_tagged']:  # Code in the tags
                tex = '\n'.join([r'\let\oldthetable=\thetable',
                                 r'\renewcommand\thetable{%s}'%\
                                 references[attrs[0]]])
                pre = RawBlock('tex', tex)
                tex = '\n'.join([r'\let\thetable=\oldthetable',
                                 r'\addtocounter{table}{-1}'])
                post = RawBlock('tex', tex)
                return [pre, AttrTable(*value), post]
        elif table['is_unreferenceable']:
            attrs[0] = ''  # The label isn't needed any further
        elif fmt in ('html', 'html5') and LABEL_PATTERN.match(attrs[0]):
            # Insert anchor
            anchor = RawBlock('html', '<a name="%s"></a>'%attrs[0])
            return [anchor, AttrTable(*value)]
        elif fmt == 'docx':
            # As per http://officeopenxml.com/WPhyperlink.php
            bookmarkstart = \
              RawBlock('openxml',
                       '<w:bookmarkStart w:id="0" w:name="%s"/>'
                       %attrs[0])
            bookmarkend = \
              RawBlock('openxml', '<w:bookmarkEnd w:id="0"/>')
            return [bookmarkstart, AttrTable(*value), bookmarkend]

    return None