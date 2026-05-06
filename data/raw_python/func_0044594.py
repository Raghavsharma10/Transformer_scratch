def naccess_available():
    """True if naccess is available on the path."""
    available = False
    try:
        subprocess.check_output(['naccess'], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        available = True
    except FileNotFoundError:
        print("naccess has not been found on your path. If you have already "
              "installed naccess but are unsure how to add it to your path, "
              "check out this: https://stackoverflow.com/a/14638025")
    return available