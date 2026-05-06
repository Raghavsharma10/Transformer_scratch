def append_all_below(destination, source, join_str=None):
    """
    Compared to xml.dom.minidom, lxml's treatment of text as .text and .tail
    attributes of elements is an oddity. It can even be a little frustrating
    when one is attempting to copy everything underneath some element to
    another element; one has to write in extra code to handle the text. This
    method provides the functionality of adding everything underneath the
    source element, in preserved order, to the destination element.
    """
    if join_str is None:
        join_str = ' '
    if source.text is not None:  # If source has text
        if len(destination) == 0:  # Destination has no children
            if destination.text is None:  # Destination has no text
                destination.text = source.text
            else:  # Destination has a text
                destination.text = join_str.join([destination.text, source.text])
        else:  # Destination has children
            #Select last child
            last = destination[-1]
            if last.tail is None:  # Last child has no tail
                last.tail = source.text
            else:  # Last child has a tail
                last.tail = join_str.join([last.tail, source.text])
    for each_child in source:
        destination.append(deepcopy(each_child))