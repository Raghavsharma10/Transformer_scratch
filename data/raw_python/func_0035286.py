def minify_print(
        ast,
        obfuscate=False,
        obfuscate_globals=False,
        shadow_funcname=False,
        drop_semi=False):
    """
    Simple minify print function; returns a string rendering of an input
    AST of an ES5 program

    Arguments

    ast
        The AST to minify print
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

    return ''.join(chunk.text for chunk in minify_printer(
        obfuscate, obfuscate_globals, shadow_funcname, drop_semi)(ast))