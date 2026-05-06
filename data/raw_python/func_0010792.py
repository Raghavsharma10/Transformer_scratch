def _insert_manifest_item(configurator, key, item):
    """ Insert an item in the list of an existing manifest key """
    with _open_manifest(configurator) as f:
        manifest = f.read()
    if item in ast.literal_eval(manifest).get(key, []):
        return
    pattern = """(["']{}["']:\\s*\\[)""".format(key)
    repl = """\\1\n        '{}',""".format(item)
    manifest = re.sub(pattern, repl, manifest, re.MULTILINE)
    with _open_manifest(configurator, "w") as f:
        f.write(manifest)