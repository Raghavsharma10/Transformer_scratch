def clean_all(ctx, dry_run=False):
    """Clean up everything, even the precious stuff.
    NOTE: clean task is executed first.
    """
    cleanup_dirs(ctx.clean_all.directories or [], dry_run=dry_run)
    cleanup_dirs(ctx.clean_all.extra_directories or [], dry_run=dry_run)
    cleanup_files(ctx.clean_all.files or [], dry_run=dry_run)
    cleanup_files(ctx.clean_all.extra_files or [], dry_run=dry_run)
    execute_cleanup_tasks(ctx, cleanup_all_tasks, dry_run=dry_run)
    clean(ctx, dry_run=dry_run)