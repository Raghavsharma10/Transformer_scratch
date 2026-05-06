def get_inst_info(qry_string):
    """Get details for instances that match the qry_string.

    Execute a query against the AWS EC2 client object, that is
    based on the contents of qry_string.

    Args:
        qry_string (str): the query to be used against the aws ec2 client.
    Returns:
        qry_results (dict): raw information returned from AWS.

    """
    qry_prefix = "EC2C.describe_instances("
    qry_real = qry_prefix + qry_string + ")"
    qry_results = eval(qry_real)     # pylint: disable=eval-used
    return qry_results