def gen_ref_docs(gen_index=False):
    # type: (int, bool) -> None
    """ Generate reference documentation for the project.

    This will use **sphinx-refdoc** to generate the source .rst files for the
    reference documentation.

    Args:
        gen_index (bool):
            Set it to **True** if you want to generate the index file with the
            list of top-level packages. This is set to default as in most cases
            you only have one package per project so you can link directly to
            that package reference (and if index were generated sphinx would
            complain about file not included in toctree).
    """
    try:
        from refdoc import generate_docs
    except ImportError as ex:
        msg = ("You need to install sphinx-refdoc if you want to generate "
               "code reference docs.")

        print(msg, file=sys.stderr)
        log.err("Exception: {}".format(ex))
        sys.exit(-1)

    pretend = context.get('pretend', False)

    docs_dir = conf.get_path('docs.path', 'docs')
    docs_ref_dir = os.path.join(docs_dir, 'ref')
    refdoc_paths = conf.get('docs.reference', [])

    if os.path.exists(docs_ref_dir):
        if not pretend:
            log.info('Removing existing reference docs')
            shutil.rmtree(docs_ref_dir)
        else:
            log.info('Would remove old reference docs')

    args = {
        'out_dir': docs_ref_dir,
        'verbose': context.get('verbose', 0),
    }

    if gen_index:
        args['gen_index'] = True

    pkg_paths = [conf.proj_path(p) for p in refdoc_paths]

    if not pretend:
        log.info('Generating reference documentation')
        generate_docs(pkg_paths, **args)
    else:
        log.info("Would generate reference docs with the following params")
        shell.cprint('<90>{}', util.yaml_dump(args).rstrip())
        shell.cprint('<90>paths:\n<34>{}', util.yaml_dump(pkg_paths).rstrip())