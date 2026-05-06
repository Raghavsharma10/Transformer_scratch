def save(ctx, dest="docs.html", format="html"):
    """Save/update docs under destination directory."""
    print("STEP: Generate docs in HTML format")
    build(ctx, builder=format)

    print("STEP: Save docs under %s/" % dest)
    source_dir = Path(ctx.config.sphinx.destdir)/format
    Path(dest).rmtree_p()
    source_dir.copytree(dest)

    # -- POST-PROCESSING: Polish up.
    for part in [ ".buildinfo", ".doctrees" ]:
        partpath = Path(dest)/part
        if partpath.isdir():
            partpath.rmtree_p()
        elif partpath.exists():
            partpath.remove_p()