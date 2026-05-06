def create_cluster(JobType=None, Resources=None, Description=None, AddressId=None, KmsKeyARN=None, RoleARN=None, SnowballType=None, ShippingOption=None, Notification=None, ForwardingAddressId=None):
    """
    Creates an empty cluster. Each cluster supports five nodes. You use the  CreateJob action separately to create the jobs for each of these nodes. The cluster does not ship until these five node jobs have been created.
    See also: AWS API Documentation
    
    Examples
    Creates an empty cluster. Each cluster supports five nodes. You use the CreateJob action separately to create the jobs for each of these nodes. The cluster does not ship until these five node jobs have been created.
    Expected Output:
    
    :example: response = client.create_cluster(
        JobType='IMPORT'|'EXPORT'|'LOCAL_USE',
        Resources={
            'S3Resources': [
                {
                    'BucketArn': 'string',
                    'KeyRange': {
                        'BeginMarker': 'string',
                        'EndMarker': 'string'
                    }
                },
            ],
            'LambdaResources': [
                {
                    'LambdaArn': 'string',
                    'EventTriggers': [
                        {
                            'EventResourceARN': 'string'
                        },
                    ]
                },
            ]
        },
        Description='string',
        AddressId='string',
        KmsKeyARN='string',
        RoleARN='string',
        SnowballType='STANDARD'|'EDGE',
        ShippingOption='SECOND_DAY'|'NEXT_DAY'|'EXPRESS'|'STANDARD',
        Notification={
            'SnsTopicARN': 'string',
            'JobStatesToNotify': [
                'New'|'PreparingAppliance'|'PreparingShipment'|'InTransitToCustomer'|'WithCustomer'|'InTransitToAWS'|'WithAWS'|'InProgress'|'Complete'|'Cancelled'|'Listing'|'Pending',
            ],
            'NotifyAll': True|False
        },
        ForwardingAddressId='string'
    )
    
    
    :type JobType: string
    :param JobType: [REQUIRED]
            The type of job for this cluster. Currently, the only job type supported for clusters is LOCAL_USE .
            

    :type Resources: dict
    :param Resources: [REQUIRED]
            The resources associated with the cluster job. These resources include Amazon S3 buckets and optional AWS Lambda functions written in the Python language.
            S3Resources (list) --An array of S3Resource objects.
            (dict) --Each S3Resource object represents an Amazon S3 bucket that your transferred data will be exported from or imported into. For export jobs, this object can have an optional KeyRange value. The length of the range is defined at job creation, and has either an inclusive BeginMarker , an inclusive EndMarker , or both. Ranges are UTF-8 binary sorted.
            BucketArn (string) --The Amazon Resource Name (ARN) of an Amazon S3 bucket.
            KeyRange (dict) --For export jobs, you can provide an optional KeyRange within a specific Amazon S3 bucket. The length of the range is defined at job creation, and has either an inclusive BeginMarker , an inclusive EndMarker , or both. Ranges are UTF-8 binary sorted.
            BeginMarker (string) --The key that starts an optional key range for an export job. Ranges are inclusive and UTF-8 binary sorted.
            EndMarker (string) --The key that ends an optional key range for an export job. Ranges are inclusive and UTF-8 binary sorted.
            
            LambdaResources (list) --The Python-language Lambda functions for this job.
            (dict) --Identifies
            LambdaArn (string) --An Amazon Resource Name (ARN) that represents an AWS Lambda function to be triggered by PUT object actions on the associated local Amazon S3 resource.
            EventTriggers (list) --The array of ARNs for S3Resource objects to trigger the LambdaResource objects associated with this job.
            (dict) --The container for the EventTriggerDefinition$EventResourceARN .
            EventResourceARN (string) --The Amazon Resource Name (ARN) for any local Amazon S3 resource that is an AWS Lambda function's event trigger associated with this job.
            
            
            

    :type Description: string
    :param Description: An optional description of this specific cluster, for example Environmental Data Cluster-01 .

    :type AddressId: string
    :param AddressId: [REQUIRED]
            The ID for the address that you want the cluster shipped to.
            

    :type KmsKeyARN: string
    :param KmsKeyARN: The KmsKeyARN value that you want to associate with this cluster. KmsKeyARN values are created by using the CreateKey API action in AWS Key Management Service (AWS KMS).

    :type RoleARN: string
    :param RoleARN: [REQUIRED]
            The RoleARN that you want to associate with this cluster. RoleArn values are created by using the CreateRole API action in AWS Identity and Access Management (IAM).
            

    :type SnowballType: string
    :param SnowballType: The type of AWS Snowball appliance to use for this cluster. Currently, the only supported appliance type for cluster jobs is EDGE .

    :type ShippingOption: string
    :param ShippingOption: [REQUIRED]
            The shipping speed for each node in this cluster. This speed doesn't dictate how soon you'll get each Snowball Edge appliance, rather it represents how quickly each appliance moves to its destination while in transit. Regional shipping speeds are as follows:
            In Australia, you have access to express shipping. Typically, appliances shipped express are delivered in about a day.
            In the European Union (EU), you have access to express shipping. Typically, Snowball Edges shipped express are delivered in about a day. In addition, most countries in the EU have access to standard shipping, which typically takes less than a week, one way.
            In India, Snowball Edges are delivered in one to seven days.
            In the US, you have access to one-day shipping and two-day shipping.
            

    :type Notification: dict
    :param Notification: The Amazon Simple Notification Service (Amazon SNS) notification settings for this cluster.
            SnsTopicARN (string) --The new SNS TopicArn that you want to associate with this job. You can create Amazon Resource Names (ARNs) for topics by using the CreateTopic Amazon SNS API action.
            You can subscribe email addresses to an Amazon SNS topic through the AWS Management Console, or by using the Subscribe AWS Simple Notification Service (SNS) API action.
            JobStatesToNotify (list) --The list of job states that will trigger a notification for this job.
            (string) --
            NotifyAll (boolean) --Any change in job state will trigger a notification for this job.
            

    :type ForwardingAddressId: string
    :param ForwardingAddressId: The forwarding address ID for a cluster. This field is not supported in most regions.

    :rtype: dict
    :return: {
        'ClusterId': 'string'
    }
    
    
    """
    pass