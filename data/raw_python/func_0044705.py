def file_cmd(context, tags, archive, bundle_name, path):
    """Add a file to a bundle."""
    bundle_obj = context.obj['db'].bundle(bundle_name)
    if bundle_obj is None:
        click.echo(click.style(f"unknown bundle: {bundle_name}", fg='red'))
        context.abort()
    version_obj = bundle_obj.versions[0]
    new_file = context.obj['db'].new_file(
        path=str(Path(path).absolute()),
        to_archive=archive,
        tags=[context.obj['db'].tag(tag_name) if context.obj['db'].tag(tag_name) else
              context.obj['db'].new_tag(tag_name) for tag_name in tags]
    )
    new_file.version = version_obj
    context.obj['db'].add_commit(new_file)
    click.echo(click.style(f"new file added: {new_file.path} ({new_file.id})", fg='green'))