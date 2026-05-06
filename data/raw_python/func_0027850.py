def serialize_to_xml(root, block):
    """
    Serialize the Peer Instruction XBlock's content to XML.

    Args:
        block (PeerInstructionXBlock): The peer instruction block to serialize.
        root (etree.Element): The XML root node to update.

    Returns:
        etree.Element

    """
    root.tag = 'ubcpi'

    if block.rationale_size is not None:
        if block.rationale_size.get('min'):
            root.set('rationale_size_min', unicode(block.rationale_size.get('min')))
        if block.rationale_size.get('max'):
            root.set('rationale_size_max', unicode(block.rationale_size['max']))

    if block.algo:
        if block.algo.get('name'):
            root.set('algorithm', block.algo.get('name'))
        if block.algo.get('num_responses'):
            root.set('num_responses', unicode(block.algo.get('num_responses')))

    display_name = etree.SubElement(root, 'display_name')
    display_name.text = block.display_name

    question = etree.SubElement(root, 'question')
    question_text = etree.SubElement(question, 'text')
    question_text.text = block.question_text['text']
    serialize_image(block.question_text, question)

    options = etree.SubElement(root, 'options')
    serialize_options(options, block)

    seeds = etree.SubElement(root, 'seeds')
    serialize_seeds(seeds, block)