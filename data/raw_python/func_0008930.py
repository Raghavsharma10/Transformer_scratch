def browse(ctx):
    """Open documentation in web browser."""
    page_html = Path(ctx.config.sphinx.destdir)/"html"/"index.html"
    if not page_html.exists():
        build(ctx, builder="html")
    assert page_html.exists()
    open_cmd = "open"   # -- WORKS ON: MACOSX
    if sys.platform.startswith("win"):
        open_cmd = "start"
    ctx.run("{open} {page_html}".format(open=open_cmd, page_html=page_html))