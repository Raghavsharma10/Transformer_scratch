def finalize_media(app):
    """Point media files at our media server."""

    if (app.builder.name == 'readthedocssinglehtmllocalmedia' or
            app.builder.format != 'html' or
            not hasattr(app.builder, 'script_files')):
        return  # Use local media for downloadable files
    # Pull project data from conf.py if it exists
    context = app.builder.config.html_context
    STATIC_URL = context.get('STATIC_URL', DEFAULT_STATIC_URL)
    js_file = '{}javascript/readthedocs-doc-embed.js'.format(STATIC_URL)
    if sphinx.version_info < (1, 8):
        app.builder.script_files.append(js_file)
    else:
        app.add_js_file(js_file)