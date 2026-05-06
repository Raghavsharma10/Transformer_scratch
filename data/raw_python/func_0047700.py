def make_dir_name_from_url(url):
    """This function attempts to emulate something like Git's "humanish"
    directory naming for clone. It's probably not a perfect facimile,
    but it's close."""
    url_path = urlparse(url).path
    head, tail = os.path.split(url_path)
    # If tail happens to be empty as in case `/foo/`, use foo.
    # If we are looking at a valid but ugly path such as
    # `/foo/.git`, use the "foo" not the ".git".
    if len(tail) is 0 or tail[0] is ".":
        head, tail = os.path.split(head)
    dir_name, ext = os.path.splitext(tail)
    return dir_name