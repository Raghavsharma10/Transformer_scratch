def copy_preset(preset_dir, project_dir):
    """Copy contents of preset into new project

    If package.json contains the key "contents", limit
    the files copied to those present in this list.

    Arguments:
        preset_dir (str): Absolute path to preset
        project_dir (str): Absolute path to new project

    """

    os.makedirs(project_dir)

    package_file = os.path.join(preset_dir, "package.json")
    with open(package_file) as f:
        package = json.load(f)

    for fname in os.listdir(preset_dir):
        src = os.path.join(preset_dir, fname)

        contents = package.get("contents") or []

        if fname not in self.files + contents:
            continue

        if os.path.isfile(src):
            shutil.copy2(src, project_dir)
        else:
            dest = os.path.join(project_dir, fname)
            shutil.copytree(src, dest)