def dump_etree_helper(container_name, data, rules, nsmap, attrib):
    """Convert DataCite JSON format to DataCite XML.

    JSON should be validated before it is given to to_xml.
    """
    output = etree.Element(container_name, nsmap=nsmap, attrib=attrib)

    for rule in rules:
        if rule not in data:
            continue

        element = rules[rule](rule, data[rule])
        for e in element:
            output.append(e)

    return output