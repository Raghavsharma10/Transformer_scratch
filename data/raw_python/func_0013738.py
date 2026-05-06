def update_job(JobId=None, RoleARN=None, Notification=None, Resources=None, AddressId=None, ShippingOption=None, Description=None, SnowballCapacityPreference=None, ForwardingAddressId=None):
    """
    While a job's JobState value is New , you can update some of the information associated with a job. Once the job changes to a different job state, usually within 60 minutes of the job being created, this action is no longer available.
    See also: AWS API Documentation
    
    Examples
    This action allows you to update certain parameters for a job. Once the job changes to a different job state, usually within 60 minutes of the job being created, this action is no longer available.
    Expected Output:
    
    :example: response = client.update_job(
        JobId='string',
        RoleARN='string',
        Notification={
            'SnsTopicARN': 'string',
            'JobStatesToNotify': [
                'New'|'PreparingAppliance'|'PreparingShipment'|'InTransitToCustomer'|'WithCustomer'|'InTransitToAWS'|'WithAWS'|'InProgress'|'Complete'|'Cancelled'|'Listing'|'Pending',
            ],
            'NotifyAll': True|False
        },
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
        AddressId='string',
        ShippingOption='SECOND_DAY'|'NEXT_DAY'|'EXPRESS'|'STANDARD',
        Description='string',
        SnowballCapacityPreference='T50'|'T80'|'T100'|'NoPreference',
        ForwardingAddressId='string'
    )
    
    
    :type JobId: string
    :param JobId: [REQUIRED]
            The job ID of the job that you want to update, for example JID123e4567-e89b-12d3-a456-426655440000 .
            

    :type RoleARN: string
    :param RoleARN: The new role Amazon Resource Name (ARN) that you want to associate with this job. To create a role ARN, use the CreateRole AWS Identity and Access Management (IAM) API action.

    :type Notification: dict
    :param Notification: The new or updated Notification object.
            SnsTopicARN (string) --The new SNS TopicArn that you want to associate with this job. You can create Amazon Resource Names (ARNs) for topics by using the CreateTopic Amazon SNS API action.
            You can subscribe email addresses to an Amazon SNS topic through the AWS Management Console, or by using the Subscribe AWS Simple Notification Service (SNS) API action.
            JobStatesToNotify (list) --The list of job states that will trigger a notification for this job.
            (string) --
            NotifyAll (boolean) --Any change in job state will trigger a notification for this job.
            

    :type Resources: dict
    :param Resources: The updated S3Resource object (for a single Amazon S3 bucket or key range), or the updated JobResource object (for multiple buckets or key ranges).
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
            
            
            

    :type AddressId: string
    :param AddressId: The ID of the updated Address object.

    :type ShippingOption: string
    :param ShippingOption: The updated shipping option value of this job's ShippingDetails object.

    :type Description: string
    :param Description: The updated description of this job's JobMetadata object.

    :type SnowballCapacityPreference: string
    :param SnowballCapacityPreference: The updated SnowballCapacityPreference of this job's JobMetadata object. The 50 TB Snowballs are only available in the US regions.

    :type ForwardingAddressId: string
    :param ForwardingAddressId: The updated ID for the forwarding address for a job. This field is not supported in most regions.

    :rtype: dict
    :return: {}
    
    
    :returns: 
    (dict) --
    
    """
    pass