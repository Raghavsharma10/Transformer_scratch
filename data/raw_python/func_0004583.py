def cc(filename: 'input source file',
       output:
       'output file name. default to be replacing input file\'s suffix with ".py"' = None,
       name: 'name of language' = 'unname'):
    """
    rbnf source code compiler.
    """

    lang = Language(name)

    with Path(filename).open('r') as fr:
        build_language(fr.read(), lang, filename)

    if not output:
        base, _ = os.path.splitext(filename)

        output = base + '.py'
    lang.dump(output)