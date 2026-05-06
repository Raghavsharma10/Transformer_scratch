def init(force):
    """Initialize registered aliases and mappings."""
    click.secho('Creating indexes...', fg='green', bold=True, file=sys.stderr)
    with click.progressbar(
            current_search.create(ignore=[400] if force else None),
            length=current_search.number_of_indexes) as bar:
        for name, response in bar:
            bar.label = name
    click.secho('Putting templates...', fg='green', bold=True, file=sys.stderr)
    with click.progressbar(
            current_search.put_templates(ignore=[400] if force else None),
            length=len(current_search.templates.keys())) as bar:
        for response in bar:
            bar.label = response