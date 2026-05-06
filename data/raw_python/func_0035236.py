def p_block(p):
    """block : '{' commands '}' """
    # section 3.2: REQUIRE command must come before any other commands,
    # which means it can't be in the block of another command
    if any(command.RULE_IDENTIFIER == 'REQUIRE'
           for command in p[2].commands):
        print("REQUIRE command not allowed inside of a block (line %d)" %
            (p.lineno(2)))
        raise SyntaxError
    p[0] = p[2]