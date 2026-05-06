def _launch_editor(starting_text=''):
    "Launch editor, let user write text, then return that text."
    # TODO: What is a reasonable default for windows? Does this approach even
    # make sense on windows?
    editor = os.environ.get('EDITOR', 'vim')

    with tempfile.TemporaryDirectory() as dirname:
        filename = pathlib.Path(dirname) / 'metadata.yml'
        with filename.open(mode='wt') as handle:
            handle.write(starting_text)
        subprocess.call([editor, filename])

        with filename.open(mode='rt') as handle:
            text = handle.read()
    return text