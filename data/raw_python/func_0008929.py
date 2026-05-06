def build(ctx, builder="html", options=""):
    """Build docs with sphinx-build"""
    sourcedir = ctx.config.sphinx.sourcedir
    destdir = Path(ctx.config.sphinx.destdir or "build")/builder
    destdir = destdir.abspath()
    with cd(sourcedir):
        destdir_relative = Path(".").relpathto(destdir)
        command = "sphinx-build {opts} -b {builder} {sourcedir} {destdir}" \
                    .format(builder=builder, sourcedir=".",
                            destdir=destdir_relative, opts=options)
        ctx.run(command)