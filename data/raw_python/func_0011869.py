def make_dir(fname):
    """
    Create the directory of a fully qualified file name if it does not exist.

    :param fname: File name
    :type  fname: string

    Equivalent to these Bash shell commands:

    .. code-block:: bash

        $ fname="${HOME}/mydir/myfile.txt"
        $ dir=$(dirname "${fname}")
        $ mkdir -p "${dir}"

    :param fname: Fully qualified file name
    :type  fname: string
    """
    file_path, fname = os.path.split(os.path.abspath(fname))
    if not os.path.exists(file_path):
        os.makedirs(file_path)