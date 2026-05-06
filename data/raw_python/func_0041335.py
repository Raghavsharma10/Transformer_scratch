def mock_lockfile_update(path):
    """
    This is a mock update. In place of this, you might simply shell out
    to a command like `yarn upgrade`.
    """
    updated_lockfile_contents = {
        'package1': '1.2.0'
    }
    with open(path, 'w+') as f:
        f.write(json.dumps(updated_lockfile_contents, indent=4))
    return updated_lockfile_contents