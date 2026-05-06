def body_json(soup, base_url=None):
    """ Get body json and then alter it with section wrapping and removing boxed-text """
    body_content = body(soup, remove_key_info_box=True, base_url=base_url)
    # Wrap in a section if the first block is not a section
    if (body_content and len(body_content) > 0 and "type" in body_content[0]
        and body_content[0]["type"] != "section"):
        # Wrap this one
        new_body_section = OrderedDict()
        new_body_section["type"] = "section"
        new_body_section["id"] = "s0"
        new_body_section["title"] = "Main text"
        new_body_section["content"] = []
        for body_block in body_content:
            new_body_section["content"].append(body_block)
        new_body = []
        new_body.append(new_body_section)
        body_content = new_body
    body_content_rewritten = elifetools.json_rewrite.rewrite_json("body_json", soup, body_content)
    return body_content_rewritten