def clean(ctx, dry_run=False):
    """Cleanup temporary dirs/files to regain a clean state."""
    # -- VARIATION-POINT 1: Allow user to override in configuration-file
    directories = ctx.clean.directories
    files = ctx.clean.files

    # -- VARIATION-POINT 2: Allow user to add more files/dirs to be removed.
    extra_directories = ctx.clean.extra_directories or []
    extra_files = ctx.clean.extra_files or []
    if extra_directories:
        directories.extend(extra_directories)
    if extra_files:
        files.extend(extra_files)

    # -- PERFORM CLEANUP:
    execute_cleanup_tasks(ctx, cleanup_tasks, dry_run=dry_run)
    cleanup_dirs(directories, dry_run=dry_run)
    cleanup_files(files, dry_run=dry_run)