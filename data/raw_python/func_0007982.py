def _make_gist(contents, description='', filename='data.geojson'):
    """
    Create and return an anonymous gist with a single file and specified
    contents

    """
    ghapi = github3.GitHub()
    files = {filename: {'content': contents}}
    gist = ghapi.create_gist(description, files)

    return gist