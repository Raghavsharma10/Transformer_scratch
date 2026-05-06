def p_command(p):
    """command : IDENTIFIER arguments ';'
               | IDENTIFIER arguments block"""
    #print("COMMAND:", p[1], p[2], p[3])
    tests = p[2].get('tests')
    block = None
    if p[3] != ';': block = p[3]
    handler = sifter.handler.get('command', p[1])
    if handler is None:
        print("No handler registered for command '%s' on line %d" %
            (p[1], p.lineno(1)))
        raise SyntaxError
    p[0] = handler(arguments=p[2]['args'], tests=tests, block=block)