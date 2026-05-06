def push_build(id, tag_prefix):
    """
    Push build to Brew
    """
    req = swagger_client.BuildRecordPushRequestRest()
    req.tag_prefix = tag_prefix
    req.build_record_id = id
    response = utils.checked_api_call(pnc_api.build_push, 'push', body=req)
    if response:
        return utils.format_json_list(response)