def build_CLASS(prefix):
    """
    Function to dowwnload CLASS from github and and build the library
    """
    # latest class version and download link
    args = (package_basedir, package_basedir, CLASS_VERSION, os.path.abspath(prefix))
    command = 'sh %s/depends/install_class.sh %s %s %s' %args

    ret = os.system(command)
    if ret != 0:
        raise ValueError("could not build CLASS v%s" %CLASS_VERSION)