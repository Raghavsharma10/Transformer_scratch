def delete_change_set(awsclient, change_set_name, stack_name):
    """Delete specified change set. Currently we only use this during
    automated regression testing. But we have plans so lets locate this
    functionality here

    :param awsclient:
    :param change_set_name:
    :param stack_name:
    """
    client = awsclient.get_client('cloudformation')

    response = client.delete_change_set(
        ChangeSetName=change_set_name,
        StackName=stack_name)