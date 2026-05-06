def main():
    """Provide the program's entry point when directly executed."""
    if len(sys.argv) != 2:
        print("Usage: {} USERNAME".format(sys.argv[0]))
        return 1

    caching_requestor = prawcore.Requestor(
        "prawcore_device_id_auth_example", session=CachingSession()
    )
    authenticator = prawcore.TrustedAuthenticator(
        caching_requestor,
        os.environ["PRAWCORE_CLIENT_ID"],
        os.environ["PRAWCORE_CLIENT_SECRET"],
    )
    authorizer = prawcore.ReadOnlyAuthorizer(authenticator)
    authorizer.refresh()

    user = sys.argv[1]
    with prawcore.session(authorizer) as session:
        data1 = session.request("GET", "/api/v1/user/{}/trophies".format(user))

    with prawcore.session(authorizer) as session:
        data2 = session.request("GET", "/api/v1/user/{}/trophies".format(user))

    for trophy in data1["data"]["trophies"]:
        description = trophy["data"]["description"]
        print(
            "Original:",
            trophy["data"]["name"]
            + (" ({})".format(description) if description else ""),
        )

    for trophy in data2["data"]["trophies"]:
        description = trophy["data"]["description"]
        print(
            "Cached:",
            trophy["data"]["name"]
            + (" ({})".format(description) if description else ""),
        )
    print(
        "----\nCached == Original:",
        data2["data"]["trophies"] == data2["data"]["trophies"],
    )

    return 0