def create_record(awsclient, name_prefix, instance_reference, type="A", host_zone_name=None):
    """
    Builds route53 record entries enabling DNS names for services
    Note: gcdt.route53 create_record(awsclient, ...)
    is used in dataplatform cloudformation.py templates!

    :param name_prefix: The sub domain prefix to use
    :param instance_reference: The EC2 troposphere reference which's private IP should be linked to
    :param type: The type of the record  A or CNAME (default: A)
    :param host_zone_name: The host zone name to use (like preprod.ds.glomex.cloud. - DO NOT FORGET THE DOT!)
    :return: RecordSetType
    """

    # Only fetch the host zone from the COPS stack if nessary
    if host_zone_name is None:
        host_zone_name = _retrieve_stack_host_zone_name(awsclient)

    if not (type == "A" or type == "CNAME"):
        raise Exception("Record set type is not supported!")

    name_of_record = name_prefix \
                         .replace('.', '') \
                         .replace('-', '') \
                         .title() + "HostRecord"

    # Reference EC2 instance automatically to their private IP
    if isinstance(instance_reference, Instance):
        resource_record = troposphere.GetAtt(
                instance_reference,
                "PrivateIp"
        )
    else:
        resource_record = instance_reference

    return RecordSetType(
            name_of_record,
            HostedZoneName=host_zone_name,
            Name=troposphere.Join("", [
                name_prefix + ".",
                host_zone_name,
            ]),
            Type=type,
            TTL=TTL_DEFAULT,
            ResourceRecords=[
                resource_record
            ],
    )