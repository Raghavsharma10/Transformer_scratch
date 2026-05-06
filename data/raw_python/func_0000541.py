def on_builder_inited(app):
    """
    Hooks into Sphinx's ``builder-inited`` event.

    Used for copying over CSS files to theme directory.
    """
    local_css_path = pathlib.Path(__file__).parent / "uqbar.css"
    theme_css_path = (
        pathlib.Path(app.srcdir) / app.config.html_static_path[0] / "uqbar.css"
    )
    with local_css_path.open("r") as file_pointer:
        local_css_contents = file_pointer.read()
    uqbar.io.write(local_css_contents, theme_css_path)