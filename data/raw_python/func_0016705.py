def find_germanet_xml_files(xml_path):
    '''
    Globs the XML files contained in the given directory and sorts
    them into sections for import into the MongoDB database.

    Arguments:
    - `xml_path`: the path to the directory containing the GermaNet
      XML files
    '''
    xml_files = sorted(glob.glob(os.path.join(xml_path, '*.xml')))

    # sort out the lexical files
    lex_files = [xml_file for xml_file in xml_files if
                 re.match(r'(adj|nomen|verben)\.',
                          os.path.basename(xml_file).lower())]
    xml_files = sorted(set(xml_files) - set(lex_files))

    if not lex_files:
        print('ERROR: cannot find lexical information files')

    # sort out the GermaNet relations file
    gn_rels_file = [xml_file for xml_file in xml_files if
                    os.path.basename(xml_file).lower() == 'gn_relations.xml']
    xml_files = sorted(set(xml_files) - set(gn_rels_file))

    if not gn_rels_file:
        print('ERROR: cannot find relations file gn_relations.xml')
        gn_rels_file = None
    else:
        if 1 < len(gn_rels_file):
            print ('WARNING: more than one relations file gn_relations.xml, '
                   'taking first match')
        gn_rels_file = gn_rels_file[0]

    # sort out the wiktionary paraphrase files
    wiktionary_files = [xml_file for xml_file in xml_files if
                        re.match(r'wiktionaryparaphrases-',
                                 os.path.basename(xml_file).lower())]
    xml_files = sorted(set(xml_files) - set(wiktionary_files))

    if not wiktionary_files:
        print('WARNING: cannot find wiktionary paraphrase files')

    # sort out the interlingual index file
    ili_files = [xml_file for xml_file in xml_files if
                os.path.basename(xml_file).lower().startswith(
            'interlingualindex')]
    xml_files = sorted(set(xml_files) - set(ili_files))

    if not ili_files:
        print('WARNING: cannot find interlingual index file')

    if xml_files:
        print('WARNING: unrecognised xml files:', xml_files)

    return lex_files, gn_rels_file, wiktionary_files, ili_files