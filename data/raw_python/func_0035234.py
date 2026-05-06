def p_commands_list(p):
    """commands : commands command"""
    p[0] = p[1]

    # section 3.2: REQUIRE command must come before any other commands
    if p[2].RULE_IDENTIFIER == 'REQUIRE':
        if any(command.RULE_IDENTIFIER != 'REQUIRE'
               for command in p[0].commands):
            print("REQUIRE command on line %d must come before any "
                  "other non-REQUIRE commands" % p.lineno(2))
            raise SyntaxError

    # section 3.1: ELSIF and ELSE must follow IF or another ELSIF
    elif p[2].RULE_IDENTIFIER in ('ELSIF', 'ELSE'):
        if p[0].commands[-1].RULE_IDENTIFIER not in ('IF', 'ELSIF'):
            print("ELSIF/ELSE command on line %d must follow an IF/ELSIF "
                  "command" % p.lineno(2))
            raise SyntaxError

    p[0].commands.append(p[2])