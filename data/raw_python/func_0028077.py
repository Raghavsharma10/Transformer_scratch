def parse_for_simple_stems(output, skip_empty=False, skip_same_stems=True):
    """
    Parses the output stem lines to produce a list with possible stems
    for each word in the output.

    :param skip_empty: set True to skip lines without stems (default is False)
    :returns: a list of tuples, each containing an original text word and
              a list of stems for the given word
    """
    lines_with_stems = _get_lines_with_stems(output)
    stems = list()

    last_word = None
    for line in lines_with_stems:
        word, stem, _ = line.split("\t")
        stem = stem if stem != '-' else None

        if skip_empty and (stem is None):
            continue

        if last_word != word:
            stems.append((word, []))

        ## append new stem only if not on list already
        stem = None if skip_same_stems and stem in stems[-1][1] else stem
        if stem is not None:
            stems[-1][1].append(stem)

        last_word = word

    return stems