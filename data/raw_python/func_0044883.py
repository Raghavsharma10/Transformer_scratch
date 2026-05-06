def save_current_nb_as_html(info=False):
    """
    Save the current notebook as html file in the same directory
    """
    assert in_ipynb()

    full_path = get_notebook_name()
    path, filename = os.path.split(full_path)

    wd_save = os.getcwd()
    os.chdir(path)
    cmd = 'jupyter nbconvert --to html "{}"'.format(filename)
    os.system(cmd)
    os.chdir(wd_save)

    if info:
        print("target dir: ", path)
        print("cmd: ", cmd)
        print("working dir: ", wd_save)