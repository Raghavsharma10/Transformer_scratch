def create_toc(headlines, hyperlink=True, top_link=False, no_toc_header=False):
    """
    Creates the table of contents from the headline list
    that was returned by the tag_and_collect function.

    Keyword Arguments:
        headlines: list of lists
            e.g., ['Some header lvl3', 'some-header-lvl3', 3]
        hyperlink: Creates hyperlinks in Markdown format if True,
            e.g., '- [Some header lvl1](#some-header-lvl1)'
        top_link: if True, add a id tag for linking the table
            of contents itself (for the back-to-top-links)
        no_toc_header: suppresses TOC header if True.

    Returns  a list of headlines for a table of contents
    in Markdown format,
    e.g., ['        - [Some header lvl3](#some-header-lvl3)', ...]

    """
    processed = []
    if not no_toc_header:
        if top_link:
            processed.append('<a class="mk-toclify" id="table-of-contents"></a>\n')
        processed.append('# Table of Contents')

    for line in headlines:
        if hyperlink:
            item = '%s- [%s](#%s)' % ((line[2]-1)*'    ', line[0], line[1])
        else:
            item = '%s- %s' % ((line[2]-1)*'    ', line[0])
        processed.append(item)
    processed.append('\n')
    return processed