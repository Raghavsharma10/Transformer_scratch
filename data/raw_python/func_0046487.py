def retrieve_author(id=None, username=None):
    """
    Retrieve a SpigotAuthor via their id, or username.
    :param id:
    :param username:
    :return:
    """

    if id is None and username is None:
        raise SpigotAuthorException("Unable to retrieve an Author without an Identifier")

    if id is None:
        return SpigotAuthor.from_username(username)
    else:
        return SpigotAuthor.from_id(id)