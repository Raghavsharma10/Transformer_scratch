def untag(name, tag_name):
    """
    Remove the given tag from the given metric.
    Return True if the metric was tagged, False otherwise
    """

    with LOCK:
        by_tag = TAGS.get(tag_name, None)
        if not by_tag:
            return False
        try:
            by_tag.remove(name)

            # remove the tag if no associations left
            if not by_tag:
                TAGS.pop(tag_name)

            return True
        except KeyError:
            return False