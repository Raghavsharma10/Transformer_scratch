def update_docs(readme=True, makefiles=True):
    """Update documentation (ready for publishing new release)

    Usually called by ``make docs``

    :param bool make_doc: generate DOC page from Makefile help messages

    """
    if readme:
        _pandoc = get_external_executable("pandoc")
        rst2markdown_github(os.path.join(_HERE, "README.rst"),
                            os.path.join(_HERE, "README.md"),
                            pandoc=_pandoc)

    if makefiles:
        _make = get_external_executable("make")
        project_makefile_dir = os.path.abspath(_HERE)
        project_makefile_rst = os.path.join(
            _HERE,
            'docs',
            'src',
            'project_makefile.rst')
        docs_makefile_dir = os.path.join(_HERE, 'docs', 'src')
        docs_makefile_rst = os.path.join(
            _HERE,
            'docs',
            'src',
            'docs_makefile.rst')

        #: ``help2rst_queue`` stores tuples of
        #: ``(cwd, help_cmd, path_to_rst_file, rst_title_of_new_file)``
        help2rst_queue = [
            (project_makefile_dir, [_make, "help"], project_makefile_rst,
             "Project ``Makefile``"),

            (docs_makefile_dir, [_make, "help"], docs_makefile_rst,
             "Documentation ``Makefile``")]

        for cwd, help_cmd, outfile, title in help2rst_queue:
            console_help2rst(
                cwd,
                help_cmd,
                outfile,
                title,
                format_as_code=True)