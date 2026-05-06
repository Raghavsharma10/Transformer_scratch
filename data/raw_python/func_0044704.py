def bundle(context, name):
    """Add a new bundle."""
    if context.obj['db'].bundle(name):
        click.echo(click.style('bundle name already exists', fg='yellow'))
        context.abort()
    new_bundle = context.obj['db'].new_bundle(name)
    context.obj['db'].add_commit(new_bundle)

    # add default version
    new_version = context.obj['db'].new_version(created_at=new_bundle.created_at)
    new_version.bundle = new_bundle
    context.obj['db'].add_commit(new_version)

    click.echo(click.style(f"new bundle added: {new_bundle.name} ({new_bundle.id})", fg='green'))