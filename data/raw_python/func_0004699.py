def run_putgist(filename, user, **kwargs):
    """Passes user inputs to GetGist() and calls put()"""
    assume_yes = kwargs.get("yes_to_all")
    private = kwargs.get("private")
    getgist = GetGist(
        user=user,
        filename=filename,
        assume_yes=assume_yes,
        create_private=private,
        allow_none=True,
    )
    getgist.put()