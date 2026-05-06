def load_stanzas(stanzas_file):
    """
    Load stanzas from gold standard file
    """
    f = stanzas_file.readlines()
    stanzas = []
    for i, line in enumerate(f):
        if i % 4 == 0:
            stanza_words = line.strip().split()[1:]
            stanzas.append(Stanza(stanza_words))
    return stanzas