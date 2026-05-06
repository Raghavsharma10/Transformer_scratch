def docs(ctx, output='html', rebuild=False, show=True, verbose=True):
    """Build the docs and show them in default web browser."""
    sphinx_build = ctx.run(
        'sphinx-build -b {output} {all} {verbose} docs docs/_build'.format(
            output=output,
            all='-a -E' if rebuild else '',
            verbose='-v' if verbose else ''))
    if not sphinx_build.ok:
        fatal("Failed to build the docs", cause=sphinx_build)

    if show:
        path = os.path.join(DOCS_OUTPUT_DIR, 'index.html')
        if sys.platform == 'darwin':
            path = 'file://%s' % os.path.abspath(path)
        webbrowser.open_new_tab(path)