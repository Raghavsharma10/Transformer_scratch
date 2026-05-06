def selector_to_text(sel, guess_punct_space=True, guess_layout=True):
    """ Convert a cleaned parsel.Selector to text.
    See html_text.extract_text docstring for description of the approach
    and options.
    """
    import parsel
    if isinstance(sel, parsel.SelectorList):
        # if selecting a specific xpath
        text = []
        for s in sel:
            extracted = etree_to_text(
                s.root,
                guess_punct_space=guess_punct_space,
                guess_layout=guess_layout)
            if extracted:
                text.append(extracted)
        return ' '.join(text)
    else:
        return etree_to_text(
            sel.root,
            guess_punct_space=guess_punct_space,
            guess_layout=guess_layout)