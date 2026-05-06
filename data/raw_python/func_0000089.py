def remove_not_allowed_in_console():
    '''This function should be called from the console, when it starts.

    Some transformers are not allowed in the console and they could have
    been loaded prior to the console being activated. We effectively remove them
    and print an information message specific to that transformer
    as written in the transformer module.

    '''
    not_allowed_in_console = []
    if CONSOLE_ACTIVE:
        for name in transformers:
            tr_module = import_transformer(name)
            if hasattr(tr_module, "NO_CONSOLE"):
                not_allowed_in_console.append((name, tr_module))
        for name, tr_module in not_allowed_in_console:
            print(tr_module.NO_CONSOLE)
            # Note: we do not remove them, so as to avoid seeing the
            # information message displayed again if an attempt is
            # made to re-import them from a console instruction.
            transformers[name] = NullTransformer()