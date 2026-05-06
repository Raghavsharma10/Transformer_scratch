def push_build_status(id):
    """
    Get status of Brew push.
    """
    response = utils.checked_api_call(pnc_api.build_push, 'status', build_record_id=id)
    if response:
        return utils.format_json(response)