def main():
    '''
    entry point of the application.

    Parses the CLI commands and runs the actions.
    '''
    args = CLI.parse_args(__doc__)

    if args['--verbose']:
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)

    if not args['-k']:
        print("No API key given. Please create an API key on <https://octopart.com/api/dashboard>")
        return ReturnValues.NO_APIKEY

    if args['-t'] == 'octopart':
        engine = PyPartsOctopart(args['-k'], verbose=args['--verbose'])
    elif args['-t'] == 'parts.io':
        engine = PyPartsPartsIO(args['-k'], verbose=args['--verbose'])
    else:
        engine = PyPartsBase(args['-k'], verbose=args['--verbose'])

    try:
        if 'lookup' in args or 'search' in args:
            return engine.part_search(args['<part>'])
        elif 'specs' in args:
            return engine.part_specs(args['<part>'])
        elif 'datasheet' in args:
            if args['<action>'] == 'open':
                if args['--output']:
                    return engine.part_datasheet(args['<part>'], command=args['--command'], path=args['--output'])
                else:
                    return engine.part_datasheet(args['<part>'], command=args['--command'])
            elif args['<action>'] == 'save':
                return engine.part_datasheet(args['<part>'], path=args['--output'])
        elif 'show' in args:
            return engine.part_show(args['<part>'], printout=args['--print'])
    except OctopartException as err:
        print(err)
        return ReturnValues.RUNTIME_ERROR