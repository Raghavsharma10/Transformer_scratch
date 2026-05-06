def nickmask(prefix: str, kwargs: Dict[str, Any]) -> None:
    """ store nick, user, host in kwargs if prefix is correct format """
    if "!" in prefix and "@" in prefix:
        # From a user
        kwargs["nick"], remainder = prefix.split("!", 1)
        kwargs["user"], kwargs["host"] = remainder.split("@", 1)
    else:
        # From a server, probably the host
        kwargs["host"] = prefix