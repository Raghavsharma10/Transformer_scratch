def body_block_paragraph_render(p_tag, html_flag=True, base_url=None):
    """
    paragraphs may wrap some other body block content
    this is separated out so it can be called from more than one place
    """
    # Configure the XML to HTML conversion preference for shorthand use below
    convert = lambda xml_string: xml_to_html(html_flag, xml_string, base_url)

    block_content_list = []

    tag_content_content = []
    nodenames = body_block_nodenames()

    paragraph_content = u''
    for child_tag in p_tag:

        if child_tag.name is None or body_block_content(child_tag) == {}:
            paragraph_content = paragraph_content + unicode_value(child_tag)

        else:
            # Add previous paragraph content first
            if paragraph_content.strip() != '':
                tag_content_content.append(body_block_paragraph_content(convert(paragraph_content)))
                paragraph_content = u''

        if child_tag.name is not None and body_block_content(child_tag) != {}:
            for block_content in body_block_content_render(child_tag, base_url=base_url):
                if block_content != {}:
                    tag_content_content.append(block_content)
    # finish up
    if paragraph_content.strip() != '':
        tag_content_content.append(body_block_paragraph_content(convert(paragraph_content)))

    if len(tag_content_content) > 0:
        for block_content in tag_content_content:
            block_content_list.append(block_content)

    return block_content_list