def regex_in_file(regex, filepath, return_match=False):
    """ Search for a regex in a file

    If return_match is True, return the found object instead of a boolean
    """
    file_content = get_file_content(filepath)
    re_method = funcy.re_find if return_match else funcy.re_test
    return re_method(regex, file_content)