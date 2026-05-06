def parse_question_xml(root):
    """
    Parse <question> element in the UBCPI XBlock's content XML.

    Args:
        root (lxml.etree.Element): The root of the <question> node in the tree.

    Returns:
        dict, a deserialized representation of a question. E.g.
        {
            'text': 'What is the answer to life, the universe and everything?',
            'image_url': '',
            'image_position': 'below',
            'image_show_fields': 0,
            'image_alt': 'description'
        }

    Raises:
        ValidationError: The XML definition is invalid.
    """
    question_dict = dict()

    question_prompt_el = root.find('text')
    if question_prompt_el is not None:
        question_dict['text'] = _safe_get_text(question_prompt_el)
    else:
        raise ValidationError(_('Question must have text element.'))

    # optional image element
    question_dict.update(parse_image_xml(root))

    return question_dict