def files(context, yes, tag, bundle, before, notondisk):
    """Delete files based on tags."""
    file_objs = []

    if not tag and not bundle:
        click.echo("I'm afraid I can't let you do that.")
        context.abort()

    if bundle:
        bundle_obj = context.obj['store'].bundle(bundle)
        if bundle_obj is None:
            click.echo(click.style('bundle not found', fg='red'))
            context.abort()

    query = context.obj['store'].files_before(bundle = bundle, tags = tag, before = before)

    if notondisk:
        file_objs = set(query) - context.obj['store'].files_ondisk(query)
    else:
        file_objs = query.all()

    if len(file_objs) > 0 and len(yes) < 2:
        if not click.confirm(f"Are you sure you want to delete {len(file_objs)} files?"):
            context.abort()

    for file_obj in file_objs:
        if yes or click.confirm(f"remove file from disk and database: {file_obj.full_path}"):
            file_obj_path = Path(file_obj.full_path)
            if file_obj.is_included and (file_obj_path.exists() or file_obj_path.is_symlink()):
                file_obj_path.unlink()
            file_obj.delete()
            context.obj['store'].commit()
            click.echo(f'{file_obj.full_path} deleted')