def body_block_paragraph_content(text):
    "for formatting of simple paragraphs of text only, and check if it is all whitespace"
    tag_content = OrderedDict()
    if text and text != '':
        tag_content["type"] = "paragraph"
        tag_content["text"] = clean_whitespace(text)
    return tag_content