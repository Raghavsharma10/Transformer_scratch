def get(context, tags: List[str], version: int, verbose: bool, bundle: str):
    """Get files."""
    store = Store(context.obj['database'], context.obj['root'])
    files = store.files(bundle=bundle, tags=tags, version=version)
    for file_obj in files:
        if verbose:
            tags = ', '.join(tag.name for tag in file_obj.tags)
            click.echo(f"{click.style(str(file_obj.id), fg='blue')} | {file_obj.full_path} | "
                       f"{click.style(tags, fg='yellow')}")
        else:
            click.echo(file_obj.full_path)