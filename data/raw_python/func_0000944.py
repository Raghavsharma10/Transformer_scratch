def body_block_caption_render(caption_tags, base_url=None):
    """fig and media tag captions are similar so use this common function"""
    caption_content = []
    supplementary_material_tags = []

    for block_tag in remove_doi_paragraph(caption_tags):
        # Note then skip p tags with supplementary-material inside
        if raw_parser.supplementary_material(block_tag):
            for supp_tag in raw_parser.supplementary_material(block_tag):
                supplementary_material_tags.append(supp_tag)
            continue

        for block_content in body_block_content_render(block_tag, base_url=base_url):

            if block_content != {}:
                caption_content.append(block_content)

    return caption_content, supplementary_material_tags