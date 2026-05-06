def clean(ctx, dry_run=False):
    """Cleanup generated document artifacts."""
    basedir = ctx.sphinx.destdir or "build/docs"
    cleanup_dirs([basedir], dry_run=dry_run)