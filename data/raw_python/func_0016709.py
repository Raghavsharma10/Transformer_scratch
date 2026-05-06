def read_paraphrase_file(filename):
    '''
    Reads in a GermaNet wiktionary paraphrase file and returns its
    contents as a list of dictionary structures.

    Arguments:
    - `filename`:
    '''
    with open(filename, 'rb') as input_file:
        doc = etree.parse(input_file)

    assert doc.getroot().tag == 'wiktionaryParaphrases'
    paraphrases = []
    for child in doc.getroot():
        if child.tag == 'wiktionaryParaphrase':
            paraphrase = child
            warn_attribs('', paraphrase, PARAPHRASE_ATTRIBS)
            if 0 < len(paraphrase):
                print('unrecognised child of <wiktionaryParaphrase>',
                      list(paraphrase))
            paraphrase_dict = dict(paraphrase.items())
            if paraphrase_dict['edited'] not in MAP_YESNO_TO_BOOL:
                print('<paraphrase> attribute "edited" has unexpected value',
                      paraphrase_dict['edited'])
            else:
                paraphrase_dict['edited'] = MAP_YESNO_TO_BOOL[
                    paraphrase_dict['edited']]
            if not paraphrase_dict['wiktionarySenseId'].isdigit():
                print('<paraphrase> attribute "wiktionarySenseId" has '
                      'non-integer value', paraphrase_dict['edited'])
            else:
                paraphrase_dict['wiktionarySenseId'] = \
                    int(paraphrase_dict['wiktionarySenseId'], 10)
            paraphrases.append(paraphrase_dict)
        else:
            print('unknown child of <wiktionaryParaphrases>', child)

    return paraphrases