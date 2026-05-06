def create_nouns(max=2):
    """
    Return a string of random nouns up to max number
    """
    nouns = []
    for noun in range(0, max):
        nouns.append(random.choice(noun_list))
    return " ".join(nouns)