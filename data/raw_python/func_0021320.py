def check_py_version():
    """Check if a propper Python version is used."""
    try:
        if sys.version_info >= (2, 7):
            return
    except:
        pass
    print(" ")
    print(" ERROR - memtop needs python version at least 2.7")
    print(("Chances are that you can install newer version from your "
           "repositories, or even that you have some newer version "
           "installed yet."))
    print("(one way to find out which versions are installed is to try "
          "following: 'which python2.7' , 'which python3' and so...)")
    print(" ")
    sys.exit(-1)