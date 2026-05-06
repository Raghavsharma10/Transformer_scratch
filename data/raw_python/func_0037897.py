def main():
    '''Main in a function in case you place a build.py for pydocs inside the root directory.'''

    options = '''
        pydocs

        Usage:
            pydocs SOURCE OUTPUT_DIR
            pydocs SOURCE OUTPUT_DIR [--json] [--index NAME] [--ignore FILE,NAMES]
            pydocs --help

        Options:
            SOURCE                  Source module, or . for current directory.
            OUTPUT_DIR              The location to output the generated markdown.
            --json                  Dump meta in JSON format upon completion.
            --index NAME            Name of the index file (default index.md) to generate.
            --ignore FILE,NAMES     Comma separated modules to ignore/skip.
            -h --help               Show this screen.
            --version               Show version.
    '''
    args = docopt(options)
    build(
        getcwd(), args['SOURCE'], args['OUTPUT_DIR'],
        json_dump=args['--json'], ignore_modules=args['--ignore'], index_filename=args['--index'] or 'index'
    )