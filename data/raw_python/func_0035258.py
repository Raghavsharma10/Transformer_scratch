def obfuscate(
        obfuscate_globals=False, shadow_funcname=False, reserved_keywords=()):
    """
    An example, barebone name obfuscation ruleset

    obfuscate_globals
        If true, identifier names on the global scope will also be
        obfuscated.  Default is False.
    shadow_funcname
        If True, obfuscated function names will be shadowed.  Default is
        False.
    reserved_keywords
        A tuple of strings that should not be generated as obfuscated
        identifiers.
    """

    def name_obfuscation_rules():
        inst = Obfuscator(
            obfuscate_globals=obfuscate_globals,
            shadow_funcname=shadow_funcname,
            reserved_keywords=reserved_keywords,
        )
        return {
            'token_handler': token_handler_unobfuscate,
            'deferrable_handlers': {
                Resolve: inst.resolve,
            },
            'prewalk_hooks': [
                inst.prewalk_hook,
            ],
        }
    return name_obfuscation_rules