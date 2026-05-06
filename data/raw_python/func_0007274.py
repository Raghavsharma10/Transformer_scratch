def build_markdown(toc_headlines, body, spacer=0, placeholder=None):
    """
    Returns a string with the Markdown output contents incl.
    the table of contents.

    Keyword arguments:
        toc_headlines: lines for the table of contents
            as created by the create_toc function.
        body: contents of the Markdown file including
            ID-anchor tags as returned by the
            tag_and_collect function.
        spacer: Adds vertical space after the table
            of contents. Height in pixels.
        placeholder: If a placeholder string is provided, the placeholder
            will be replaced by the TOC instead of inserting the TOC at
            the top of the document

    """
    if spacer:
        spacer_line = ['\n<div style="height:%spx;"></div>\n' % (spacer)]
        toc_markdown = "\n".join(toc_headlines + spacer_line)
    else:
        toc_markdown = "\n".join(toc_headlines)

    body_markdown = "\n".join(body).strip()

    if placeholder:
        markdown = body_markdown.replace(placeholder, toc_markdown)
    else:
        markdown = toc_markdown + body_markdown

    return markdown