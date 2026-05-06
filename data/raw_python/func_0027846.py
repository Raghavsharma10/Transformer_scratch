def parse_seeds_xml(root):
    """
    Parse <seeds> element in the UBCPI XBlock's content XML.

    Args:
        root (lxml.etree.Element): The root of the <seeds> node in the tree.

    Returns:
        a list of deserialized representation of seeds. E.g.
        [{
            'answer': 1,  # option index starting from one
            'rationale': 'This is a seeded answer',
        },
        {....
        }]

    Raises:
        ValidationError: The XML definition is invalid.
    """
    seeds = []

    for seed_el in root.findall('seed'):
        seed_dict = dict()
        seed_dict['rationale'] = _safe_get_text(seed_el)

        if 'option' in seed_el.attrib:
            seed_dict['answer'] = int(seed_el.attrib['option']) - 1
        else:
            raise ValidationError(_('Seed element must have an option attribute.'))

        seeds.append(seed_dict)

    return seeds