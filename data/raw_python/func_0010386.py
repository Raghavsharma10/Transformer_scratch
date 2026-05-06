def split_text(text, include_part_of_speech=False, strip_english=False, strip_numbers=False):
    u"""
    Split Chinese text at word boundaries.

    include_pos: also returns the Part Of Speech for each of the words.
    Some of the different parts of speech are:
        r: pronoun
        v: verb
        ns: proper noun
        etc...

    This all gets returned as a tuple:
        index 0: the split word
        index 1: the word's part of speech

    strip_english: remove all entries that have English or numbers in them (useful sometimes)
    """

    if not include_part_of_speech:
        seg_list = pseg.cut(text)
        if strip_english:
            seg_list = filter(lambda x: not contains_english(x), seg_list)
        if strip_numbers:
            seg_list = filter(lambda x: not _is_number(x), seg_list)
        return list(map(lambda i: i.word, seg_list))
    else:
        seg_list = pseg.cut(text)
        objs = map(lambda w: (w.word, w.flag), seg_list)
        if strip_english:
            objs = filter(lambda x: not contains_english(x[0]), objs)
        if strip_english:
            objs = filter(lambda x: not _is_number(x[0]), objs)
        return objs

    # if was_traditional:
    #   seg_list = map(tradify, seg_list)

    return list(seg_list)