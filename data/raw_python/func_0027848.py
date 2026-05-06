def serialize_options(options, block):
    """
    Serialize the options in peer instruction XBlock to xml

    Args:
        options (lxml.etree.Element): The <options> XML element.
        block (PeerInstructionXBlock): The XBlock with configuration to serialize.

    Returns:
        None
    """
    for index, option_dict in enumerate(block.options):
        option = etree.SubElement(options, 'option')
        # set correct option and rationale
        if index == block.correct_answer:
            option.set('correct', u'True')

            if hasattr(block, 'correct_rationale'):
                rationale = etree.SubElement(option, 'rationale')
                rationale.text = block.correct_rationale['text']

        text = etree.SubElement(option, 'text')
        text.text = option_dict.get('text', '')

        serialize_image(option_dict, option)