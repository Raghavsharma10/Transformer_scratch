def split_hostmask(hostmask):
    """Splits a nick@host string into nick and host."""
    nick, _, host = hostmask.partition('@')
    nick, _, user = nick.partition('!')
    return nick, user or None, host or None