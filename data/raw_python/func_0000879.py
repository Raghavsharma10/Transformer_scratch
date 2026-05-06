def rewrite_elife_body_json(json_content, doi):
    """ rewrite elife body json """

    # Edge case add an id to a section
    if doi == "10.7554/eLife.00013":
        if (json_content and len(json_content) > 0):
            if (json_content[0].get("type") and json_content[0].get("type") == "section"
                and json_content[0].get("title") and json_content[0].get("title") =="Introduction"
                and not json_content[0].get("id")):
                json_content[0]["id"] = "s1"

    # Edge case remove an extra section
    if doi == "10.7554/eLife.04232":
        if (json_content and len(json_content) > 0):
            for outer_block in json_content:
                if outer_block.get("id") and outer_block.get("id") == "s4":
                    for mid_block in outer_block.get("content"):
                        if mid_block.get("id") and mid_block.get("id") == "s4-6":
                            for inner_block in mid_block.get("content"):
                                if inner_block.get("content") and not inner_block.get("title"):
                                    mid_block["content"] = inner_block.get("content")

    # Edge case remove unwanted sections
    if doi == "10.7554/eLife.04871":
        if (json_content and len(json_content) > 0):
            for i, outer_block in enumerate(json_content):
                if (outer_block.get("id") and outer_block.get("id") in ["s7", "s8"]
                    and not outer_block.get("title")):
                    if outer_block.get("content"):
                        json_content[i] = outer_block.get("content")[0]

    # Edge case remove an extra section
    if doi == "10.7554/eLife.05519":
        if (json_content and len(json_content) > 0):
            for outer_block in json_content:
                if outer_block.get("id") and outer_block.get("id") == "s4":
                    for mid_block in outer_block.get("content"):
                        if mid_block.get("content") and not mid_block.get("id"):
                            new_blocks = []
                            for inner_block in mid_block.get("content"):
                                 new_blocks.append(inner_block)
                            outer_block["content"] = new_blocks

    # Edge case add a title to a section
    if doi == "10.7554/eLife.07157":
        if (json_content and len(json_content) > 0):
            if (json_content[0].get("type") and json_content[0].get("type") == "section"
                and json_content[0].get("id") and json_content[0].get("id") == "s1"):
                json_content[0]["title"] = "Main text"

    # Edge case remove a section with no content
    if doi == "10.7554/eLife.09977":
        if (json_content and len(json_content) > 0):
            i_index = j_index = None
            for i, outer_block in enumerate(json_content):
                if (outer_block.get("id") and outer_block.get("id") == "s4"
                    and outer_block.get("content")):
                    # We have i
                    i_index = i
                    break
            if i_index is not None:
                for j, inner_block in enumerate(json_content[i_index].get("content")):
                    if (inner_block.get("id") and inner_block.get("id") == "s4-11"
                        and inner_block.get("content") is None):
                        # Now we have i and j for deletion outside of the loop
                        j_index = j
                        break
            # Do the deletion on the original json
            if i_index is not None and j_index is not None:
                del json_content[i_index]["content"][j_index]

    # Edge case wrap sections differently
    if doi == "10.7554/eLife.12844":
        if (json_content and len(json_content) > 0 and json_content[0].get("type")
            and json_content[0]["type"] == "section"):
            new_body = OrderedDict()
            for i, tag_block in enumerate(json_content):
                if i == 0:
                    tag_block["title"] = "Main text"
                    new_body = tag_block
                elif i > 0:
                    new_body["content"].append(tag_block)
            json_content = [new_body]

    return json_content