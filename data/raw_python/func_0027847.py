def parse_from_xml(root):
    """
    Update the UBCPI XBlock's content from an XML definition.

    We need to be strict about the XML we accept, to avoid setting
    the XBlock to an invalid state (which will then be persisted).

    Args:
        root (lxml.etree.Element): The XML definition of the XBlock's content.

    Returns:
        A dictionary of all of the XBlock's content.

    Raises:
        UpdateFromXmlError: The XML definition is invalid
    """

    # Check that the root has the correct tag
    if root.tag != 'ubcpi':
        raise UpdateFromXmlError(_('Every peer instruction tool must contain an "ubcpi" element.'))

    display_name_el = root.find('display_name')
    if display_name_el is None:
        raise UpdateFromXmlError(_('Every peer instruction tool must contain a "display_name" element.'))
    else:
        display_name = _safe_get_text(display_name_el)

    rationale_size_min = int(root.attrib['rationale_size_min']) if 'rationale_size_min' in root.attrib else None
    rationale_size_max = int(root.attrib['rationale_size_max']) if 'rationale_size_max' in root.attrib else None

    question_el = root.find('question')
    if question_el is None:
        raise UpdateFromXmlError(_('Every peer instruction must tool contain a "question" element.'))
    else:
        question = parse_question_xml(question_el)

    options_el = root.find('options')
    if options_el is None:
        raise UpdateFromXmlError(_('Every peer instruction must tool contain a "options" element.'))
    else:
        options, correct_answer, correct_rationale = parse_options_xml(options_el)

    seeds_el = root.find('seeds')
    if seeds_el is None:
        raise UpdateFromXmlError(_('Every peer instruction must tool contain a "seeds" element.'))
    else:
        seeds = parse_seeds_xml(seeds_el)

    algo = unicode(root.attrib['algorithm']) if 'algorithm' in root.attrib else None
    num_responses = unicode(root.attrib['num_responses']) if 'num_responses' in root.attrib else None

    return {
        'display_name': display_name,
        'question_text': question,
        'options': options,
        'rationale_size': {'min': rationale_size_min, 'max': rationale_size_max},
        'correct_answer': correct_answer,
        'correct_rationale': correct_rationale,
        'seeds': seeds,
        'algo': {"name": algo, 'num_responses': num_responses}
    }