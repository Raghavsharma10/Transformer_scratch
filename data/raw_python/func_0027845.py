def parse_options_xml(root):
    """
    Parse <options> element in the UBCPI XBlock's content XML.

    Args:
        root (lxml.etree.Element): The root of the <options> node in the tree.

    Returns:
        a list of deserialized representation of options. E.g.
        [{
            'text': 'Option 1',
            'image_url': '',
            'image_position': 'below',
            'image_show_fields': 0,
            'image_alt': ''
        },
        {....
        }]

    Raises:
        ValidationError: The XML definition is invalid.
    """
    options = []
    correct_option = None
    rationale = None

    for option_el in root.findall('option'):
        option_dict = dict()
        option_prompt_el = option_el.find('text')
        if option_prompt_el is not None:
            option_dict['text'] = _safe_get_text(option_prompt_el)
        else:
            raise ValidationError(_('Option must have text element.'))

        # optional image element
        option_dict.update(parse_image_xml(option_el))

        if 'correct' in option_el.attrib and _parse_boolean(option_el.attrib['correct']):
            if correct_option is None:
                correct_option = len(options)
                rationale_el = option_el.find('rationale')
                if rationale_el is not None:
                    rationale = {'text': _safe_get_text(rationale_el)}
                else:
                    raise ValidationError(_('Missing rationale for correct answer.'))
            else:
                raise ValidationError(_('Only one correct answer can be defined in options.'))

        options.append(option_dict)

    if correct_option is None or rationale is None:
        raise ValidationError(_('Correct answer and rationale are required and have to be defined in one of the option.'))

    return options, correct_option, rationale