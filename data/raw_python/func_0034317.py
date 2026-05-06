def get_user_data_triplet(username, password):
    """Print out user data triplet: username, password verifier, salt."""
    context = SRPContext(username, password)
    username, password_verifier, salt = context.get_user_data_triplet(base64=True)

    click.secho('Username: %s' % username)
    click.secho('Password verifier: %s' % password_verifier)
    click.secho('Salt: %s' % salt)