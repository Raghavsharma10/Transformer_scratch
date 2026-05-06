def lex(args):
    """ Lex input and return a list of actions to perform. """
    if len(args) == 0 or args[0] == SHOW:
        return [(SHOW, None)]
    elif args[0] == LOG:
        return [(LOG, None)]
    elif args[0] == ECHO:
        return [(ECHO, None)]
    elif args[0] == SET and args[1] == RATE:
        return tokenizeSetRate(args[2:])
    elif args[0] == SET and args[1] == DAYS:
        return tokenizeSetDays(args[2:])
    elif args[0] == TAKE:
        return tokenizeTake(args[1:])
    elif args[0] == CANCEL:
        return tokenizeCancel(args[1:])
    elif isMonth(args[0]):
        return tokenizeTake(args)
    else:
        print('Unknown commands: {}'.format(' '.join(args)))
        return []