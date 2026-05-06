def remove_preset(preset):
    """Physically delete local preset

    Arguments:
        preset (str): Name of preset

    """

    preset_dir = os.path.join(presets_dir(), preset)

    try:
        shutil.rmtree(preset_dir)
    except IOError:
        lib.echo("\"%s\" did not exist" % preset)