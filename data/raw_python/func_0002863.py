def append_new_text(destination, text, join_str=None):
    """
    This method provides the functionality of adding text appropriately
    underneath the destination node. This will be either to the destination's
    text attribute or to the tail attribute of the last child.
    """
    if join_str is None:
        join_str = ' '
    if len(destination) > 0:  # Destination has children
        last = destination[-1]
        if last.tail is None:  # Last child has no tail
            last.tail = text
        else:  # Last child has a tail
            last.tail = join_str.join([last.tail, text])
    else:  # Destination has no children
        if destination.text is None:  # Destination has no text
            destination.text = text
        else:  # Destination has a text
            destination.text = join_str.join([destination.text, text])