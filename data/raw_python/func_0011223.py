def random_name():
    """Return a random name

    Example:
        >> random_name()
        dizzy_badge
        >> random_name()
        evasive_cactus

    """

    adj = _data.adjectives[random.randint(0, len(_data.adjectives) - 1)]
    noun = _data.nouns[random.randint(0, len(_data.nouns) - 1)]
    return "%s_%s" % (adj, noun)