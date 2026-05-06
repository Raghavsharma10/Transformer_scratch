def body_block_title_label_caption(tag_content, title_value, label_value,
                                   caption_content, set_caption=True, prefer_title=False, prefer_label=False):
    """set the title, label and caption values in a consistent way

    set_caption: insert a "caption" field
    prefer_title: when only one value is available, set title rather than label. If False, set label rather than title"""
    set_if_value(tag_content, "label", rstrip_punctuation(label_value))
    set_if_value(tag_content, "title", title_value)
    if set_caption is True and caption_content and len(caption_content) > 0:
        tag_content["caption"] = caption_content
    if prefer_title:
        if "title" not in tag_content and label_value:
            set_if_value(tag_content, "title", label_value)
            del(tag_content["label"])
    if prefer_label:
        if "label" not in tag_content and title_value:
            set_if_value(tag_content, "label", rstrip_punctuation(title_value))
            del(tag_content["title"])