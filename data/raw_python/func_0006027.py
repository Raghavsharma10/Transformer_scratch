def read_html_file(filename):
    """Reads the contents of an html file in the css directory

    @return: Contents of the specified file
    """
    with open(os.path.join(get_static_directory(), 'html/{filename}'.format(filename=filename))) as f:
        contents = f.read()
    return contents