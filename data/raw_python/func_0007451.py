def add_helpingmaterials(config, helping_materials_file, helping_type):
    """Add helping materials to a project."""
    res = _add_helpingmaterials(config, helping_materials_file, helping_type)
    click.echo(res)