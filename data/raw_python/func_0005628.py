def extract_changelog_items(text, tags):
    # type: (str) -> Dict[str, List[str]]
    """ Extract all tagged items from text.

    Args:
        text (str):
            Text to extract the tagged items from. Each tagged item is a
            paragraph that starts with a tag. It can also be a text list item.

    Returns:
        tuple[list[str], list[str], list[str]]:
            A tuple of `(features, changes, fixes)` extracted from the given
            text.

    The tagged items are usually features/changes/fixes but it can be configured
    through `pelconf.yaml`.
    """

    patterns = {x['header']: tag_re(x['tag']) for x in tags}
    items = {x['header']: [] for x in tags}
    curr_tag = None
    curr_text = ''

    for line in text.splitlines():
        if not line.strip():
            if curr_tag is not None:
                items[curr_tag].append(curr_text)
                curr_text = ''
            curr_tag = None

        for tag in tags:
            m = patterns[tag['header']].match(line)
            if m:
                if curr_tag is not None:
                    items[curr_tag].append(curr_text)
                    curr_text = ''

                curr_tag = tag['header']
                line = m.group('text')
                break

        if curr_tag is not None:
            curr_text = '{} {}'.format(curr_text.strip(), line.strip()).strip()

    if curr_tag is not None:
        items[curr_tag].append(curr_text)

    return items