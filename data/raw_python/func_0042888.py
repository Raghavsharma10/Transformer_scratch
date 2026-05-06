def register_from_options(options=None, template=None, extractor=None):
    """Register the spec codec using the provided options"""
    if template is None:
        from noseOfYeti.plugins.support.spec_options import spec_options as template

    if extractor is None:
        from noseOfYeti.plugins.support.spec_options import extract_options_dict as extractor

    config = Config(template)
    config.setup(options, extractor)

    imports = determine_imports(
          extra_imports = ';'.join([d for d in config.extra_import if d])
        , with_default_imports = config.with_default_imports
        )

    tok = Tokeniser(
          default_kls = config.default_kls
        , import_tokens = imports
        , wrapped_setup = config.wrapped_setup
        , with_describe_attrs = not config.no_describe_attrs
        )

    TokeniserCodec(tok).register()