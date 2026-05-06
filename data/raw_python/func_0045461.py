async def get_user(username: str, api_key: str, **kwargs) -> User:
    """ Creates a new user, validate its credentials and returns it

    |funccoro|

    Args:
        username: username as specified on the challonge website
        api_key: key as found on the challonge
            `settings <https://challonge.com/settings/developer>`_

    Returns:
        User: a logged in user if no exception has been raised

    Raises:
        APIException

    """
    new_user = User(username, api_key, **kwargs)
    await new_user.validate()
    return new_user