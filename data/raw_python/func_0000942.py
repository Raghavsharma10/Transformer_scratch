def body_block_content_render(tag, recursive=False, base_url=None):
    """
    Render the tag as body content and call recursively if
    the tag has child tags
    """
    block_content_list = []
    tag_content = OrderedDict()

    if tag.name == "p":
        for block_content in body_block_paragraph_render(tag, base_url=base_url):
            if block_content != {}:
                block_content_list.append(block_content)
    else:
        tag_content = body_block_content(tag, base_url=base_url)

    nodenames = body_block_nodenames()

    tag_content_content = []

    # Collect the content of the tag but only for some tags
    if tag.name not in ["p", "fig", "table-wrap", "list", "media", "disp-quote", "code"]:
        for child_tag in tag:
            if not(hasattr(child_tag, 'name')):
                continue

            if child_tag.name == "p":
                # Ignore paragraphs that start with DOI:
                if node_text(child_tag) and len(remove_doi_paragraph([child_tag])) <= 0:
                    continue
                for block_content in body_block_paragraph_render(child_tag, base_url=base_url):
                    if block_content != {}:
                        tag_content_content.append(block_content)

            elif child_tag.name == "fig" and tag.name == "fig-group":
                # Do not fig inside fig-group a second time
                pass
            elif child_tag.name == "media" and tag.name == "fig-group":
                # Do not include a media video inside fig-group a second time
                if child_tag.get("mimetype") == "video":
                    pass
            else:
                for block_content in body_block_content_render(child_tag, recursive=True, base_url=base_url):
                    if block_content != {}:
                        tag_content_content.append(block_content)

    if len(tag_content_content) > 0:
        if tag.name in nodenames or recursive is False:
            tag_content["content"] = []
            for block_content in tag_content_content:
                tag_content["content"].append(block_content)
            block_content_list.append(tag_content)
        else:
            # Not a block tag, e.g. a caption tag, let the content pass through
            block_content_list = tag_content_content
    else:
        block_content_list.append(tag_content)

    return block_content_list