def on_line(client, line):
    """Default handling for incoming lines.

    This handler will automatically manage the following IRC messages:

      PING:
        Responds with a PONG.
      PRIVMSG:
        Dispatches the PRIVMSG event.
      NOTICE:
        Dispatches the NOTICE event.
      MOTDSTART:
        Initializes MOTD receive buffer.
      MOTD:
        Appends a line to the MOTD receive buffer.
      ENDOFMOTD:
        Joins the contents of the MOTD receive buffer, assigns the result
        to the .motd of the server, and dispatches the MOTD event.
    """
    if line.startswith("PING"):
        client.send("PONG" + line[4:])
        return True

    if line.startswith(":"):
        actor, _, line = line[1:].partition(" ")
    else:
        actor = None
    command, _, args = line.partition(" ")
    command = NUMERIC_EVENTS.get(command, command)

    parser = PARSERS.get(command, False)
    if parser:
        parser(client, command, actor, args)
        return True
    elif parser is False:
        # Explicitly ignored message
        return True