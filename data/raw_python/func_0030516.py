def build(id=None, name=None, revision=None,
          temporary_build=False, timestamp_alignment=False,
          no_build_dependencies=False,
          keep_pod_on_failure=False,
          force_rebuild=False,
          rebuild_mode=common.REBUILD_MODES_DEFAULT):
    """
    Trigger a BuildConfiguration by name or ID
    """
    data = build_raw(id, name, revision, temporary_build, timestamp_alignment, no_build_dependencies,
              keep_pod_on_failure, force_rebuild, rebuild_mode)
    if data:
        return utils.format_json(data)