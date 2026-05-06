def body_block_supplementary_material_render(supp_tags, base_url=None):
    """fig and media tag caption may have supplementary material"""
    source_data = []
    for supp_tag in supp_tags:
        for block_content in body_block_content_render(supp_tag, base_url=base_url):
            if block_content != {}:
                if "content" in block_content:
                    del block_content["content"]
                source_data.append(block_content)
    return source_data