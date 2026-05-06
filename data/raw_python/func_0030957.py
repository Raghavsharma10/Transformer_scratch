def update(taxids, conn, force_download, silent):
    """Update local UniProt database"""
    if not silent:
        click.secho("WARNING: Update is very time consuming and can take several "
                    "hours depending which organisms you are importing!", fg="yellow")

        if not taxids:
            click.echo("Please note that you can restrict import to organisms by "
                       "NCBI taxonomy IDs")
            click.echo("Example (human, mouse, rat):\n")
            click.secho("\tpyuniprot update --taxids 9606,10090,10116\n\n", fg="green")

    if taxids:
        taxids = [int(taxid.strip()) for taxid in taxids.strip().split(',') if re.search('^ *\d+ *$', taxid)]

    database.update(taxids=taxids, connection=conn, force_download=force_download, silent=silent)