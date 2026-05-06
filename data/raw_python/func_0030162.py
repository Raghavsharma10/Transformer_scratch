def build_set(id=None, name=None, temporary_build=False, timestamp_alignment=False,
              force=False, rebuild_mode=common.REBUILD_MODES_DEFAULT, **kwargs):
    """
    Start a build of the given BuildConfigurationSet
    """
    content = build_set_raw(id, name,
                            temporary_build, timestamp_alignment, force, rebuild_mode, **kwargs)
    if content:
        return utils.format_json(content)