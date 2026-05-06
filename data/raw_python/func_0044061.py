def bundle(context, yes, bundle_name):
    """Delete the latest bundle version."""
    bundle_obj = context.obj['store'].bundle(bundle_name)
    if bundle_obj is None:
        click.echo(click.style('bundle not found', fg='red'))
        context.abort()
    version_obj = bundle_obj.versions[0]
    if version_obj.included_at:
        question = f"remove bundle version from file system and database: {version_obj.full_path}"
    else:
        question = f"remove bundle version from database: {version_obj.created_at.date()}"
    if yes or click.confirm(question):
        if version_obj.included_at:
            shutil.rmtree(version_obj.full_path, ignore_errors=True)
        version_obj.delete()
        context.obj['store'].commit()
        click.echo(f"version deleted: {version_obj.full_path}")