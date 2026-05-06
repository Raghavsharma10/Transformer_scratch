def build_set_raw(id=None, name=None,
                  tempbuild=False, timestamp_alignment=False,
                  force=False, rebuild_mode=common.REBUILD_MODES_DEFAULT,
                  **kwargs):
    """
    Start a build of the given BuildConfigurationSet
    """
    logging.debug("temp_build: " + str(tempbuild))
    logging.debug("timestamp_alignment: " + str(timestamp_alignment))
    logging.debug("force: " + str(force))
    if tempbuild is False and timestamp_alignment is True:
        logging.error("You can only activate timestamp alignment with the temporary build flag!")
        sys.exit(1)

    found_id = common.set_id(pnc_api.build_group_configs, id, name)

    revisions = kwargs.get("id_revisions")
    if revisions:
        id_revs = map(__parse_revision, revisions)

        bcsRest = common.get_entity(pnc_api.build_group_configs, found_id)
        body = swagger_client.BuildConfigurationSetWithAuditedBCsRest()
        body = __fill_BCSWithAuditedBCs_body(body, bcsRest, id_revs)

        response = utils.checked_api_call(pnc_api.build_group_configs, 'build_versioned', id=found_id,
                                          temporary_build=tempbuild,
                                          timestamp_alignment=timestamp_alignment,
                                          force_rebuild=force,
                                          rebuild_mode=rebuild_mode,
                                          body=body)
    else:
        response = utils.checked_api_call(pnc_api.build_group_configs, 'build', id=found_id,
                                      temporary_build=tempbuild,
                                      timestamp_alignment=timestamp_alignment,
                                      force_rebuild=force,
                                      rebuild_mode=rebuild_mode)
    if response:
        return response.content