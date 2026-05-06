def boxed_text_to_image_block(tag):
    "covert boxed-text to an image block containing an inline-graphic"
    tag_block = OrderedDict()
    image_content = body_block_image_content(first(raw_parser.inline_graphic(tag)))
    tag_block["type"] = "image"
    set_if_value(tag_block, "doi", doi_uri_to_doi(object_id_doi(tag, tag.name)))
    set_if_value(tag_block, "id", tag.get("id"))
    set_if_value(tag_block, "image", image_content)
    # render paragraphs into a caption
    p_tags = raw_parser.paragraph(tag)
    caption_content = []
    for p_tag in p_tags:
        if not raw_parser.inline_graphic(p_tag):
            caption_content.append(body_block_content(p_tag))
    set_if_value(tag_block, "caption", caption_content)
    return tag_block