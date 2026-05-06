def get_private_and_public(username, password_verifier, private, preset):
    """Print out server public and private."""
    session = SRPServerSession(
        SRPContext(username, prime=preset[0], generator=preset[1]),
        hex_from_b64(password_verifier), private=private)

    click.secho('Server private: %s' % session.private_b64)
    click.secho('Server public: %s' % session.public_b64)