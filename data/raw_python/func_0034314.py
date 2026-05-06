def get_session_data( username, password_verifier, salt, client_public, private, preset):
    """Print out server session data."""
    session = SRPServerSession(
        SRPContext(username, prime=preset[0], generator=preset[1]),
        hex_from_b64(password_verifier), private=private)

    session.process(client_public, salt, base64=True)

    click.secho('Server session key: %s' % session.key_b64)
    click.secho('Server session key proof: %s' % session.key_proof_b64)
    click.secho('Server session key hash: %s' % session.key_proof_hash_b64)