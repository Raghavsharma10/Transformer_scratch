def config_examples(dest, user_dir):
    """ Copy the example workflows to a directory.

    \b
    DEST: Path to which the examples should be copied.
    """
    examples_path = Path(lightflow.__file__).parents[1] / 'examples'
    if examples_path.exists():
        dest_path = Path(dest).resolve()
        if not user_dir:
            dest_path = dest_path / 'examples'

        if dest_path.exists():
            if not click.confirm('Directory already exists. Overwrite existing files?',
                                 default=True, abort=True):
                return
        else:
            dest_path.mkdir()

        for example_file in examples_path.glob('*.py'):
            shutil.copy(str(example_file), str(dest_path / example_file.name))
        click.echo('Copied examples to {}'.format(str(dest_path)))
    else:
        click.echo('The examples source path does not exist')