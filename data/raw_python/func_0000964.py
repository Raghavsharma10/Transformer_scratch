def unwrap_appendix_box(json_content):
    """for use in removing unwanted boxed-content from appendices json"""
    if json_content.get("content") and len(json_content["content"]) > 0:
        first_block = json_content["content"][0]
        if (first_block.get("type")
            and first_block.get("type") == "box"
            and first_block.get("content")):
            if first_block.get("doi") and not json_content.get("doi"):
                json_content["doi"] = first_block.get("doi")
            json_content["content"] = first_block["content"]
    return json_content