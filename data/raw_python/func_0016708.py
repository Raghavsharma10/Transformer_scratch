def read_relation_file(filename):
    '''
    Reads the GermaNet relation file ``gn_relations.xml`` which lists
    all the relations holding between lexical units and synsets.

    Arguments:
    - `filename`:
    '''
    with open(filename, 'rb') as input_file:
        doc = etree.parse(input_file)

    lex_rels = []
    con_rels = []
    assert doc.getroot().tag == 'relations'
    for child in doc.getroot():
        if child.tag == 'lex_rel':
            if 0 < len(child):
                print('<lex_rel> has unexpected child node')
            child_dict = dict(child.items())
            warn_attribs('', child, RELATION_ATTRIBS, RELATION_ATTRIBS_REQD)
            if child_dict['dir'] not in LEX_REL_DIRS:
                print('unrecognized <lex_rel> dir', child_dict['dir'])
            if child_dict['dir'] == 'both' and 'inv' not in child_dict:
                print('<lex_rel> has dir=both but does not specify inv')
            lex_rels.append(child_dict)
        elif child.tag == 'con_rel':
            if 0 < len(child):
                print('<con_rel> has unexpected child node')
            child_dict = dict(child.items())
            warn_attribs('', child, RELATION_ATTRIBS, RELATION_ATTRIBS_REQD)
            if child_dict['dir'] not in CON_REL_DIRS:
                print('unrecognised <con_rel> dir', child_dict['dir'])
            if (child_dict['dir'] in ['both', 'revert'] and
                'inv' not in child_dict):
                print('<con_rel> has dir={0} but does not specify inv'.format(
                    child_dict['dir']))
            con_rels.append(child_dict)
        else:
            print('unrecognised child of <relations>', child)
            continue

    return lex_rels, con_rels