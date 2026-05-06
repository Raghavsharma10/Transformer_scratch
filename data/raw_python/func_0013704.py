def describe_cases(caseIdList=None, displayId=None, afterTime=None, beforeTime=None, includeResolvedCases=None, nextToken=None, maxResults=None, language=None, includeCommunications=None):
    """
    Returns a list of cases that you specify by passing one or more case IDs. In addition, you can filter the cases by date by setting values for the afterTime and beforeTime request parameters. You can set values for the includeResolvedCases and includeCommunications request parameters to control how much information is returned.
    Case data is available for 12 months after creation. If a case was created more than 12 months ago, a request for data might cause an error.
    The response returns the following in JSON format:
    See also: AWS API Documentation
    
    
    :example: response = client.describe_cases(
        caseIdList=[
            'string',
        ],
        displayId='string',
        afterTime='string',
        beforeTime='string',
        includeResolvedCases=True|False,
        nextToken='string',
        maxResults=123,
        language='string',
        includeCommunications=True|False
    )
    
    
    :type caseIdList: list
    :param caseIdList: A list of ID numbers of the support cases you want returned. The maximum number of cases is 100.
            (string) --
            

    :type displayId: string
    :param displayId: The ID displayed for a case in the AWS Support Center user interface.

    :type afterTime: string
    :param afterTime: The start date for a filtered date search on support case communications. Case communications are available for 12 months after creation.

    :type beforeTime: string
    :param beforeTime: The end date for a filtered date search on support case communications. Case communications are available for 12 months after creation.

    :type includeResolvedCases: boolean
    :param includeResolvedCases: Specifies whether resolved support cases should be included in the DescribeCases results. The default is false .

    :type nextToken: string
    :param nextToken: A resumption point for pagination.

    :type maxResults: integer
    :param maxResults: The maximum number of results to return before paginating.

    :type language: string
    :param language: The ISO 639-1 code for the language in which AWS provides support. AWS Support currently supports English ('en') and Japanese ('ja'). Language parameters must be passed explicitly for operations that take them.

    :type includeCommunications: boolean
    :param includeCommunications: Specifies whether communications should be included in the DescribeCases results. The default is true .

    :rtype: dict
    :return: {
        'cases': [
            {
                'caseId': 'string',
                'displayId': 'string',
                'subject': 'string',
                'status': 'string',
                'serviceCode': 'string',
                'categoryCode': 'string',
                'severityCode': 'string',
                'submittedBy': 'string',
                'timeCreated': 'string',
                'recentCommunications': {
                    'communications': [
                        {
                            'caseId': 'string',
                            'body': 'string',
                            'submittedBy': 'string',
                            'timeCreated': 'string',
                            'attachmentSet': [
                                {
                                    'attachmentId': 'string',
                                    'fileName': 'string'
                                },
                            ]
                        },
                    ],
                    'nextToken': 'string'
                },
                'ccEmailAddresses': [
                    'string',
                ],
                'language': 'string'
            },
        ],
        'nextToken': 'string'
    }
    
    
    :returns: 
    caseIdList (list) -- A list of ID numbers of the support cases you want returned. The maximum number of cases is 100.
    
    (string) --
    
    
    displayId (string) -- The ID displayed for a case in the AWS Support Center user interface.
    afterTime (string) -- The start date for a filtered date search on support case communications. Case communications are available for 12 months after creation.
    beforeTime (string) -- The end date for a filtered date search on support case communications. Case communications are available for 12 months after creation.
    includeResolvedCases (boolean) -- Specifies whether resolved support cases should be included in the  DescribeCases results. The default is false .
    nextToken (string) -- A resumption point for pagination.
    maxResults (integer) -- The maximum number of results to return before paginating.
    language (string) -- The ISO 639-1 code for the language in which AWS provides support. AWS Support currently supports English ("en") and Japanese ("ja"). Language parameters must be passed explicitly for operations that take them.
    includeCommunications (boolean) -- Specifies whether communications should be included in the  DescribeCases results. The default is true .
    
    """
    pass