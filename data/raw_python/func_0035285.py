def minify_printer(
        obfuscate=False,
        obfuscate_globals=False,
        shadow_funcname=False,
        drop_semi=False):
    """
    Construct a minimum printer.

    Arguments

    obfuscate
        If True, obfuscate identifiers nested in each scope with a
        shortened identifier name to further reduce output size.

        Defaults to False.
    obfuscate_globals
        Also do the same to identifiers nested on the global scope; do
        not enable unless the renaming of global variables in a not
        fully deterministic manner into something else is guaranteed to
        not cause problems with the generated code and other code that
        in the same environment that it will be executed in.

        Defaults to False for the reason above.
    drop_semi
        Drop semicolons whenever possible (e.g. the final semicolons of
        a given block).
    """

    active_rules = [rules.minify(drop_semi=drop_semi)]
    if obfuscate:
        active_rules.append(rules.obfuscate(
            obfuscate_globals=obfuscate_globals,
            shadow_funcname=shadow_funcname,
            reserved_keywords=(Lexer.keywords_dict.keys())
        ))
    return Unparser(rules=active_rules)