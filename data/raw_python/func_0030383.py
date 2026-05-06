def push_build_set(id, tag_prefix):
    """
    Push build set to Brew
    """
    req = swagger_client.BuildConfigSetRecordPushRequestRest()
    req.tag_prefix = tag_prefix
    req.build_config_set_record_id = id
    response = utils.checked_api_call(pnc_api.build_push, 'push_record_set', body=req)
    if response:
        return utils.format_json_list(response)