def on_connect(client):
    """Default on-connect actions."""
    client.nick(client.user.nick)
    client.userinfo(client.user.username, client.user.realname)