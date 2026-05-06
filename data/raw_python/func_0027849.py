def serialize_seeds(seeds, block):
    """
    Serialize the seeds in peer instruction XBlock to xml

    Args:
        seeds (lxml.etree.Element): The <seeds> XML element.
        block (PeerInstructionXBlock): The XBlock with configuration to serialize.

    Returns:
        None
    """
    for seed_dict in block.seeds:
        seed = etree.SubElement(seeds, 'seed')
        # options in xml starts with 1
        seed.set('option', unicode(seed_dict.get('answer', 0) + 1))
        seed.text = seed_dict.get('rationale', '')