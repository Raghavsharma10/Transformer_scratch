def from_etree(
    el, node=None, node_cls=None,
    tagsub=functools.partial(re.sub, r'\{.+?\}', ''),
    Node=Node):
    '''Convert the element tree to a tater tree.
    '''
    node_cls = node_cls or Node
    if node is None:
        node = node_cls()
    tag = tagsub(el.tag)
    attrib = dict((tagsub(k), v) for (k, v) in el.attrib.items())
    node.update(attrib, tag=tag)

    if el.text:
        node['text'] = el.text
    for child in el:
        child = from_etree(child, node_cls=node_cls)
        node.append(child)
    if el.tail:
        node['tail'] = el.tail
    return node