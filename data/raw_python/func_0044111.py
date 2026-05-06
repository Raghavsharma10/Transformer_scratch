def startstop(inst_id, cmdtodo):
    """Start or Stop the Specified Instance.

    Args:
        inst_id (str): instance-id to perform command against
        cmdtodo (str): command to perform (start or stop)
    Returns:
        response (dict): reponse returned from AWS after
                         performing specified action.

    """
    tar_inst = EC2R.Instance(inst_id)
    thecmd = getattr(tar_inst, cmdtodo)
    response = thecmd()
    return response