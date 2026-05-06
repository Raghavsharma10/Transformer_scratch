def is_admin(controller, client, actor):
    """Used to determine whether someone issuing a command is an admin.

    By default, checks to see if there's a line of the type nick=host that
    matches the command's actor in the [admins] section of the config file,
    or a key that matches the entire mask (e.g. "foo@bar" or "foo@bar=1").
    """
    config = controller.config
    if not config.has_section("admins"):
        logging.debug("Ignoring is_admin check - no [admins] config found.")
        return False
    for key,val in config.items("admins"):
        if actor == User(key):
            logging.debug("is_admin: %r matches admin %r", actor, key)
            return True
        if actor.nick.lower() == key.lower() and actor.host.lower() == val.lower():
            logging.debug("is_admin: %r matches admin %r=%r", actor, key, val)
            return True
    logging.debug("is_admin: %r is not an admin.", actor)
    return False