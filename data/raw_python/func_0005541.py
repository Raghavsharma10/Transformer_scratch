def docs(recreate, gen_index, run_doctests):
    # type: (bool, bool, bool) -> None
    """ Build the documentation for the project.

    Args:
        recreate (bool):
            If set to **True**, the build and output directories will be cleared
            prior to generating the docs.
        gen_index (bool):
            If set to **True**, it will generate top-level index file for the
            reference documentation.
        run_doctests (bool):
            Set to **True** if you want to run doctests after the documentation
            is generated.
        pretend (bool):
            If set to **True**, do not actually execute any shell commands, just
            print the command that would be executed.
    """
    build_dir = conf.get_path('build_dir', '.build')
    docs_dir = conf.get_path('docs.path', 'docs')
    refdoc_paths = conf.get('docs.reference', [])

    docs_html_dir = conf.get_path('docs.out', os.path.join(docs_dir, 'html'))
    docs_tests_dir = conf.get_path('docs.tests_out',
                                   os.path.join(docs_dir, 'doctest'))
    docs_build_dir = os.path.join(build_dir, 'docs')

    if recreate:
        for path in (docs_html_dir, docs_build_dir):
            if os.path.exists(path):
                log.info("<91>Deleting <94>{}".format(path))
                shutil.rmtree(path)

    if refdoc_paths:
        gen_ref_docs(gen_index)
    else:
        log.err('Not generating any reference documentation - '
                'No docs.reference specified in config')

    with conf.within_proj_dir(docs_dir):
        log.info('Building docs')
        shell.run('sphinx-build -b html -d {build} {docs} {out}'.format(
            build=docs_build_dir,
            docs=docs_dir,
            out=docs_html_dir,
        ))

        if run_doctests:
            log.info('Running doctests')
            shell.run('sphinx-build -b doctest -d {build} {docs} {out}'.format(
                build=docs_build_dir,
                docs=docs_dir,
                out=docs_tests_dir,
            ))

        log.info('You can view the docs by browsing to <34>file://{}'.format(
            os.path.join(docs_html_dir, 'index.html')
        ))