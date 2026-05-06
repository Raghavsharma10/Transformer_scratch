def check_text_similarity(a_dom, b_dom, cutoff):
    """Check whether two dom trees have similar text or not."""
    a_words = list(tree_words(a_dom))
    b_words = list(tree_words(b_dom))

    sm = WordMatcher(a=a_words, b=b_words)
    if sm.text_ratio() >= cutoff:
        return True
    return False