def pprint(string, token=[WORD, POS, CHUNK, PNP], column=4):
    """ Pretty-prints the output of Parser.parse() as a table with outlined columns.
        Alternatively, you can supply a tree.Text or tree.Sentence object.
    """
    if isinstance(string, basestring):
        print("\n\n".join([table(sentence, fill=column) for sentence in Text(string, token)]))
    if isinstance(string, Text):
        print("\n\n".join([table(sentence, fill=column) for sentence in string]))
    if isinstance(string, Sentence):
        print(table(string, fill=column))