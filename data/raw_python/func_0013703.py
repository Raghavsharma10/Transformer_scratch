def create_case(subject=None, serviceCode=None, severityCode=None, categoryCode=None, communicationBody=None, ccEmailAddresses=None, language=None, issueType=None, attachmentSetId=None):
    """
    Creates a new case in the AWS Support Center. This operation is modeled on the behavior of the AWS Support Center Create Case page. Its parameters require you to specify the following information:
    A successful  CreateCase request returns an AWS Support case number. Case numbers are used by the  DescribeCases operation to retrieve existing AWS Support cases.
    See also: AWS API Documentation
    
    
    :example: response = client.create_case(
        subject='string',
        serviceCode='string',
        severityCode='string',
        categoryCode='string',
        communicationBody='string',
        ccEmailAddresses=[
            'string',
        ],
        language='string',
        issueType='string',
        attachmentSetId='string'
    )
    
    
    :type subject: string
    :param subject: [REQUIRED]
            The title of the AWS Support case.
            

    :type serviceCode: string
    :param serviceCode: The code for the AWS service returned by the call to DescribeServices .

    :type severityCode: string
    :param severityCode: The code for the severity level returned by the call to DescribeSeverityLevels .
            Note
            The availability of severity levels depends on each customer's support subscription. In other words, your subscription may not necessarily require the urgent level of response time.
            

    :type categoryCode: string
    :param categoryCode: The category of problem for the AWS Support case.

    :type communicationBody: string
    :param communicationBody: [REQUIRED]
            The communication body text when you create an AWS Support case by calling CreateCase .
            

    :type ccEmailAddresses: list
    :param ccEmailAddresses: A list of email addresses that AWS Support copies on case correspondence.
            (string) --
            

    :type language: string
    :param language: The ISO 639-1 code for the language in which AWS provides support. AWS Support currently supports English ('en') and Japanese ('ja'). Language parameters must be passed explicitly for operations that take them.

    :type issueType: string
    :param issueType: The type of issue for the case. You can specify either 'customer-service' or 'technical.' If you do not indicate a value, the default is 'technical.'

    :type attachmentSetId: string
    :param attachmentSetId: The ID of a set of one or more attachments for the case. Create the set by using AddAttachmentsToSet .

    :rtype: dict
    :return: {
        'caseId': 'string'
    }
    
    
    :returns: 
    subject (string) -- [REQUIRED]
    The title of the AWS Support case.
    
    serviceCode (string) -- The code for the AWS service returned by the call to  DescribeServices .
    severityCode (string) -- The code for the severity level returned by the call to  DescribeSeverityLevels .
    
    Note
    The availability of severity levels depends on each customer's support subscription. In other words, your subscription may not necessarily require the urgent level of response time.
    
    
    categoryCode (string) -- The category of problem for the AWS Support case.
    communicationBody (string) -- [REQUIRED]
    The communication body text when you create an AWS Support case by calling  CreateCase .
    
    ccEmailAddresses (list) -- A list of email addresses that AWS Support copies on case correspondence.
    
    (string) --
    
    
    language (string) -- The ISO 639-1 code for the language in which AWS provides support. AWS Support currently supports English ("en") and Japanese ("ja"). Language parameters must be passed explicitly for operations that take them.
    issueType (string) -- The type of issue for the case. You can specify either "customer-service" or "technical." If you do not indicate a value, the default is "technical."
    attachmentSetId (string) -- The ID of a set of one or more attachments for the case. Create the set by using  AddAttachmentsToSet .
    
    """
    pass