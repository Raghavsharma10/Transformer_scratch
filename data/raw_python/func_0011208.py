def github_presets():
    """Return remote presets hosted on GitHub"""
    addr = ("https://raw.githubusercontent.com"
            "/mottosso/be-presets/master/presets.json")
    response = get(addr)

    if response.status_code == 404:
        lib.echo("Could not connect with preset database")
        sys.exit(lib.PROGRAM_ERROR)

    return dict((package["name"], package["repository"])
                for package in response.json().get("presets"))