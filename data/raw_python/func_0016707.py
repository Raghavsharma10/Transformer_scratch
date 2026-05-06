def read_lexical_file(filename):
    '''
    Reads in a GermaNet lexical information file and returns its
    contents as a list of dictionary structures.

    Arguments:
    - `filename`: the name of the XML file to read
    '''
    with open(filename, 'rb') as input_file:
        doc = etree.parse(input_file)

    synsets = []
    assert doc.getroot().tag == 'synsets'
    for synset in doc.getroot():
        if synset.tag != 'synset':
            print('unrecognised child of <synsets>', synset)
            continue
        synset_dict = dict(synset.items())
        synloc = '{0} synset {1},'.format(filename,
                                          synset_dict.get('id', '???'))
        warn_attribs(synloc, synset, SYNSET_ATTRIBS)
        synset_dict['lexunits'] = []
        synsets.append(synset_dict)

        for child in synset:
            if child.tag == 'lexUnit':
                lexunit      = child
                lexunit_dict = dict(lexunit.items())
                lexloc       = synloc + ' lexUnit {0},'.format(
                    lexunit_dict.get('id', '???'))
                warn_attribs(lexloc, lexunit, LEXUNIT_ATTRIBS)
                # convert some properties to booleans
                for key in ['styleMarking', 'artificial', 'namedEntity']:
                    if key in lexunit_dict:
                        if lexunit_dict[key] not in MAP_YESNO_TO_BOOL:
                            print(lexloc, ('lexunit property {0} has '
                                           'non-boolean value').format(key),
                                  lexunit_dict[key])
                            continue
                        lexunit_dict[key] = MAP_YESNO_TO_BOOL[lexunit_dict[key]]
                # convert sense to integer number
                if 'sense' in lexunit_dict:
                    if lexunit_dict['sense'].isdigit():
                        lexunit_dict['sense'] = int(lexunit_dict['sense'], 10)
                    else:
                        print(lexloc,
                              'lexunit property sense has non-numeric value',
                              lexunit_dict['sense'])
                synset_dict['lexunits'].append(lexunit_dict)
                lexunit_dict['examples'] = []
                lexunit_dict['frames']   = []
                for child in lexunit:
                    if child.tag in ['orthForm',
                                     'orthVar',
                                     'oldOrthForm',
                                     'oldOrthVar']:
                        warn_attribs(lexloc, child, set())
                        if not child.text:
                            print(lexloc, '{0} with no text'.format(child.tag))
                            continue
                        if child.tag in lexunit_dict:
                            print(lexloc, 'more than one {0}'.format(child.tag))
                        lexunit_dict[child.tag] = str(child.text)
                    elif child.tag == 'example':
                        example = child
                        text = [child for child in example
                                if child.tag == 'text']
                        if len(text) != 1 or not text[0].text:
                            print(lexloc, '<example> tag without text')
                        example_dict = {'text': str(text[0].text)}
                        for child in example:
                            if child.tag == 'text':
                                continue
                            elif child.tag == 'exframe':
                                if 'exframe' in example_dict:
                                    print(lexloc,
                                          'more than one <exframe> '
                                          'for <example>')
                                warn_attribs(lexloc, child, set())
                                if not child.text:
                                    print(lexloc, '<exframe> with no text')
                                    continue
                                example_dict['exframe'] = str(child.text)
                            else:
                                print(lexloc,
                                      'unrecognised child of <example>',
                                      child)
                        lexunit_dict['examples'].append(example_dict)
                    elif child.tag == 'frame':
                        frame = child
                        warn_attribs(lexloc, frame, set())
                        if 0 < len(frame):
                            print(lexloc, 'unrecognised <frame> children',
                                list(frame))
                        if not frame.text:
                            print(lexloc, '<frame> without text')
                            continue
                        lexunit_dict['frames'].append(str(frame.text))
                    elif child.tag == 'compound':
                        compound = child
                        warn_attribs(lexloc, compound, set())
                        compound_dict = {}
                        for child in compound:
                            if child.tag == 'modifier':
                                modifier_dict = dict(child.items())
                                warn_attribs(lexloc, child,
                                             MODIFIER_ATTRIBS, set())
                                if not child.text:
                                    print(lexloc, 'modifier without text')
                                    continue
                                modifier_dict['text'] = str(child.text)
                                if 'modifier' not in compound_dict:
                                    compound_dict['modifier'] = []
                                compound_dict['modifier'].append(modifier_dict)
                            elif child.tag == 'head':
                                head_dict = dict(child.items())
                                warn_attribs(lexloc, child, HEAD_ATTRIBS, set())
                                if not child.text:
                                    print(lexloc, '<head> without text')
                                    continue
                                head_dict['text'] = str(child.text)
                                if 'head' in compound_dict:
                                    print(lexloc,
                                          'more than one head in <compound>')
                                compound_dict['head'] = head_dict
                            else:
                                print(lexloc,
                                      'unrecognised child of <compound>',
                                      child)
                                continue
                    else:
                        print(lexloc, 'unrecognised child of <lexUnit>', child)
                        continue
            elif child.tag == 'paraphrase':
                paraphrase = child
                warn_attribs(synloc, paraphrase, set())
                paraphrase_text = str(paraphrase.text)
                if not paraphrase_text:
                    print(synloc, 'WARNING: <paraphrase> tag with no text')
            else:
                print(synloc, 'unrecognised child of <synset>', child)
                continue

    return synsets