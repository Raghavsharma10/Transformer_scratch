def main():
    """Provide the program's entry point when directly executed."""
    authenticator = prawcore.TrustedAuthenticator(
        prawcore.Requestor("prawcore_script_auth_example"),
        os.environ["PRAWCORE_CLIENT_ID"],
        os.environ["PRAWCORE_CLIENT_SECRET"],
    )
    authorizer = prawcore.ScriptAuthorizer(
        authenticator,
        os.environ["PRAWCORE_USERNAME"],
        os.environ["PRAWCORE_PASSWORD"],
    )
    authorizer.refresh()

    with prawcore.session(authorizer) as session:
        data = session.request("GET", "/api/v1/me/friends")

    for friend in data["data"]["children"]:
        print(friend["name"])

    return 0