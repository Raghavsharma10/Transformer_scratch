def etree_to_dict(source):
    """ Recursively load dict/list representation of an XML tree into an etree representation.

        Args:
            source -- An etree Element or ElementTree.

        Returns:
            A dictionary representing sorce's xml structure where tags with multiple identical childrens
            contain list of all their children dictionaries..

    >>> etree_to_dict(ET.fromstring('<content><id>12</id><title/></content>'))
    {'content': {'id': '12', 'title': None}}

    >>> etree_to_dict(ET.fromstring('<content><list><li>foo</li><li>bar</li></list></content>'))
    {'content': {'list': [{'li': 'foo'}, {'li': 'bar'}]}}
    """
    def etree_to_dict_recursive(parent):
        children = parent.getchildren()
        if children:
            d = {}
            identical_children = False
            for child in children:
                if not identical_children:
                    if child.tag in d:
                        identical_children = True
                        l = [{key: d[key]} for key in d]
                        l.append({child.tag: etree_to_dict_recursive(child)})
                        del d
                    else:
                        d.update({child.tag: etree_to_dict_recursive(child)})
                else:
                    l.append({child.tag: etree_to_dict_recursive(child)})
            return (d if not identical_children else l)
        else:
            return parent.text

    if hasattr(source, 'getroot'):
        source = source.getroot()
    if hasattr(source, 'tag'):
        return {source.tag: etree_to_dict_recursive(source)}
    else:
        raise TypeError("Requires an Element or an ElementTree.")