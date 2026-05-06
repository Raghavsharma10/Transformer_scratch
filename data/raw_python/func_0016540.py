def build_readme(base_path=None):
    """Call the conversion routine on README.md to generate README.rst.
    Why do all this? Because pypi requires reStructuredText, but markdown
    is friendlier to work with and is nicer for GitHub."""
    if base_path:
        path = os.path.join(base_path, 'README.md')
    else:
        path = 'README.md'
    convert_md_to_rst(path)
    print("Successfully converted README.md to README.rst")