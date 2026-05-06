def serialize_action(action_type, payload, **extra_fields):
    """
        This function returns the conventional form of the actions.
    """
    action_dict =  dict(
        action_type=action_type,
        payload=payload,
        **extra_fields
    )
    # return a serializable version
    return json.dumps(action_dict)