def get_private_and_public(ctx, username, password, private, preset):
    """Print out server public and private."""
    session = SRPClientSession(
        SRPContext(username, password, prime=preset[0], generator=preset[1]),
        private=private)

    click.secho('Client private: %s' % session.private_b64)
    click.secho('Client public: %s' % session.public_b64)