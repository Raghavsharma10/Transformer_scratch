def sam_cmd(sock, line, parse=True):
    """Send a line to the SAM controller, returning the parsed response"""
    sam_send(sock, line)
    reply_line = sam_readline(sock)
    if parse:
        return sam_parse_reply(reply_line)
    else:
        return reply_line