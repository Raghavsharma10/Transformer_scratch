def metasay(ctx, inputfile, item):
    """Moo some dataset metadata to stdout.

    Python module: rio-metasay
    (https://github.com/sgillies/rio-plugin-example).
    """
    with rasterio.open(inputfile) as src:
        meta = src.profile
    click.echo(moothedata(meta, key=item))