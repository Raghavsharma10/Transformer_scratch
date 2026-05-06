def create_job(JobType=None, Resources=None, Description=None, AddressId=None, KmsKeyARN=None, RoleARN=None, SnowballCapacityPreference=None, ShippingOption=None, Notification=None, ClusterId=None, SnowballType=None, ForwardingAddressId=None):
    """
    Creates a job to import or export data between Amazon S3 and your on-premises data center. Your AWS account must have the right trust policies and permissions in place to create a job for Snowball. If you're creating a job for a node in a cluster, you only need to provide the clusterId value; the other job attributes are inherited from the cluster.
    See also: AWS API Documentation
    
    Examples
    Creates a job to import or export data between Amazon S3 and your on-premises data center. Your AWS account must have the right trust policies and permissions in place to create a job for Snowball. If you're creating a job for a node in a cluster, you only need to provide the clusterId value; the other job attributes are inherited from the cluster.
    Expected Output:
    
    :example: response = client.create_job(
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
        SnowballCapacityPreference='T50'|'T80'|'T100'|'NoPreference',
        ShippingOption='SECOND_DAY'|'NEXT_DAY'|'EXPRESS'|'STANDARD',
        Notification={
            'SnsTopicARN': 'string',
            'JobStatesToNotify': [
                'New'|'PreparingAppliance'|'PreparingShipment'|'InTransitToCustomer'|'WithCustomer'|'InTransitToAWS'|'WithAWS'|'InProgress'|'Complete'|'Cancelled'|'Listing'|'Pending',
            ],
            'NotifyAll': True|False
        },
        ClusterId='string',
        SnowballType='STANDARD'|'EDGE',
        ForwardingAddressId='string'
    )
    
    
    :type JobType: string
    :param JobType: Defines the type of job that you're creating.

    :type Resources: dict
    :param Resources: Defines the Amazon S3 buckets associated with this job.
            With IMPORT jobs, you specify the bucket or buckets that your transferred data will be imported into.
            With EXPORT jobs, you specify the bucket or buckets that your transferred data will be exported from. Optionally, you can also specify a KeyRange value. If you choose to export a range, you define the length of the range by providing either an inclusive BeginMarker value, an inclusive EndMarker value, or both. Ranges are UTF-8 binary sorted.
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
    :param Description: Defines an optional description of this specific job, for example Important Photos 2016-08-11 .

    :type AddressId: string
    :param AddressId: The ID for the address that you want the Snowball shipped to.

    :type KmsKeyARN: string
    :param KmsKeyARN: The KmsKeyARN that you want to associate with this job. KmsKeyARN s are created using the CreateKey AWS Key Management Service (KMS) API action.

    :type RoleARN: string
    :param RoleARN: The RoleARN that you want to associate with this job. RoleArn s are created using the CreateRole AWS Identity and Access Management (IAM) API action.

    :type SnowballCapacityPreference: string
    :param SnowballCapacityPreference: If your job is being created in one of the US regions, you have the option of specifying what size Snowball you'd like for this job. In all other regions, Snowballs come with 80 TB in storage capacity.

    :type ShippingOption: string
    :param ShippingOption: The shipping speed for this job. This speed doesn't dictate how soon you'll get the Snowball, rather it represents how quickly the Snowball moves to its destination while in transit. Regional shipping speeds are as follows:
            In Australia, you have access to express shipping. Typically, Snowballs shipped express are delivered in about a day.
            In the European Union (EU), you have access to express shipping. Typically, Snowballs shipped express are delivered in about a day. In addition, most countries in the EU have access to standard shipping, which typically takes less than a week, one way.
            In India, Snowballs are delivered in one to seven days.
            In the US, you have access to one-day shipping and two-day shipping.
            

    :type Notification: dict
    :param Notification: Defines the Amazon Simple Notification Service (Amazon SNS) notification settings for this job.
            SnsTopicARN (string) --The new SNS TopicArn that you want to associate with this job. You can create Amazon Resource Names (ARNs) for topics by using the CreateTopic Amazon SNS API action.
            You can subscribe email addresses to an Amazon SNS topic through the AWS Management Console, or by using the Subscribe AWS Simple Notification Service (SNS) API action.
            JobStatesToNotify (list) --The list of job states that will trigger a notification for this job.
            (string) --
            NotifyAll (boolean) --Any change in job state will trigger a notification for this job.
            

    :type ClusterId: string
    :param ClusterId: The ID of a cluster. If you're creating a job for a node in a cluster, you need to provide only this clusterId value. The other job attributes are inherited from the cluster.

    :type SnowballType: string
    :param SnowballType: The type of AWS Snowball appliance to use for this job. Currently, the only supported appliance type for cluster jobs is EDGE .

    :type ForwardingAddressId: string
    :param ForwardingAddressId: The forwarding address ID for a job. This field is not supported in most regions.

    :rtype: dict
    :return: {
        'JobId': 'string'
    }
    
    
    """
    pass