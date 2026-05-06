def is_op(call, op):
    """
    :param call: The specific operator instance (a method call)
    :param op: The the operator we are testing against
    :return: isinstance(call, op), but faster
    """
    try:
        return call.id == op.id
    except Exception as e:
        return False