def tag_and_collect(lines, id_tag=True, back_links=False, exclude_h=None, remove_dashes=False):
    """
    Gets headlines from the markdown document and creates anchor tags.

    Keyword arguments:
        lines: a list of sublists where every sublist
            represents a line from a Markdown document.
        id_tag: if true, creates inserts a the <a id> tags (not req. by GitHub)
        back_links: if true, adds "back to top" links below each headline
        exclude_h: header levels to exclude. E.g., [2, 3]
            excludes level 2 and 3 headings.

    Returns a tuple of 2 lists:
        1st list:
            A modified version of the input list where
            <a id="some-header"></a> anchor tags where inserted
            above the header lines (if github is False).

        2nd list:
            A list of 3-value sublists, where the first value
            represents the heading, the second value the string
            that was inserted assigned to the IDs in the anchor tags,
            and the third value is an integer that reprents the headline level.
            E.g.,
            [['some header lvl3', 'some-header-lvl3', 3], ...]

    """
    out_contents = []
    headlines = []
    for l in lines:
        saw_headline = False

        orig_len = len(l)
        l = l.lstrip()

        if l.startswith(('# ', '## ', '### ', '#### ', '##### ', '###### ')):

            # comply with new markdown standards

            # not a headline if '#' not followed by whitespace '##no-header':
            if not l.lstrip('#').startswith(' '):
                continue
            # not a headline if more than 6 '#':
            if len(l) - len(l.lstrip('#')) > 6:
                continue
            # headers can be indented by at most 3 spaces:
            if orig_len - len(l) > 3:
                continue

            # ignore empty headers
            if not set(l) - {'#', ' '}:
                continue

            saw_headline = True
            slugified = slugify_headline(l, remove_dashes)

            if not exclude_h or not slugified[-1] in exclude_h:
                if id_tag:
                    id_tag = '<a class="mk-toclify" id="%s"></a>'\
                              % (slugified[1])
                    out_contents.append(id_tag)
                headlines.append(slugified)

        out_contents.append(l)
        if back_links and saw_headline:
            out_contents.append('[[back to top](#table-of-contents)]')
    return out_contents, headlines