def ext(obj, ext_name, ext_args):
    """ Run an extension by its name.

    \b
    EXT_NAME: The name of the extension.
    EXT_ARGS: Arguments that are passed to the extension.
    """
    try:
        mod = import_module('lightflow_{}.__main__'.format(ext_name))
        mod.main(ext_args)
    except ImportError as err:
        click.echo(_style(obj['show_color'],
                          'An error occurred when trying to call the extension',
                          fg='red', bold=True))
        click.echo('{}'.format(err))