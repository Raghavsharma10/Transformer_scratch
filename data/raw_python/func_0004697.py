def run_getgist(filename, user, **kwargs):
    """Passes user inputs to GetGist() and calls get()"""
    assume_yes = kwargs.get("yes_to_all")
    getgist = GetGist(user=user, filename=filename, assume_yes=assume_yes)
    getgist.get()