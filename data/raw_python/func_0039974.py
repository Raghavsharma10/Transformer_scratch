def cli():
    """\
    Frogsay generates an ASCII picture of a FROG spouting a FROG tip.

    FROG tips are fetched from frog.tips's API endpoint when needed,
    otherwise they are cached locally in an application-specific folder.
    """
    with open_client(cache_dir=get_cache_dir()) as client:
        tip = client.frog_tip()

    terminal_width = click.termui.get_terminal_size()[0]
    wisdom = make_frog_fresco(tip, width=terminal_width)

    click.echo(wisdom)