def output_notebook(
        d3js_url="//d3js.org/d3.v3.min",
        requirejs_url="//cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js",
        html_template=None
):
    """ Import required Javascript libraries to Jupyter Notebook. """

    if html_template is None:
        html_template = read_lib('html', 'setup')

    setup_html = populate_template(
        html_template,
        d3js=d3js_url,
        requirejs=requirejs_url
    )

    display_html(setup_html)
    return