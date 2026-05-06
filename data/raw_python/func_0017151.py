def limits(args):
    """
    Describe limits in effect on your AWS account. See also https://console.aws.amazon.com/ec2/v2/home#Limits:
    """
    # https://aws.amazon.com/about-aws/whats-new/2014/06/19/amazon-ec2-service-limits-report-now-available/
    # Console-only APIs: getInstanceLimits, getAccountLimits, getAutoscalingLimits, getHostLimits
    # http://boto3.readthedocs.io/en/latest/reference/services/dynamodb.html#DynamoDB.Client.describe_limits
    attrs = ["max-instances", "vpc-max-security-groups-per-interface", "vpc-max-elastic-ips"]
    table = clients.ec2.describe_account_attributes(AttributeNames=attrs)["AccountAttributes"]
    page_output(tabulate(table, args))