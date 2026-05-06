def create_qualification_type(Name=None, Keywords=None, Description=None, QualificationTypeStatus=None, RetryDelayInSeconds=None, Test=None, AnswerKey=None, TestDurationInSeconds=None, AutoGranted=None, AutoGrantedValue=None):
    """
    The CreateQualificationType operation creates a new Qualification type, which is represented by a QualificationType data structure.
    See also: AWS API Documentation
    
    
    :example: response = client.create_qualification_type(
        Name='string',
        Keywords='string',
        Description='string',
        QualificationTypeStatus='Active'|'Inactive',
        RetryDelayInSeconds=123,
        Test='string',
        AnswerKey='string',
        TestDurationInSeconds=123,
        AutoGranted=True|False,
        AutoGrantedValue=123
    )
    
    
    :type Name: string
    :param Name: [REQUIRED]
            The name you give to the Qualification type. The type name is used to represent the Qualification to Workers, and to find the type using a Qualification type search. It must be unique across all of your Qualification types.
            

    :type Keywords: string
    :param Keywords: One or more words or phrases that describe the Qualification type, separated by commas. The keywords of a type make the type easier to find during a search.

    :type Description: string
    :param Description: [REQUIRED]
            A long description for the Qualification type. On the Amazon Mechanical Turk website, the long description is displayed when a Worker examines a Qualification type.
            

    :type QualificationTypeStatus: string
    :param QualificationTypeStatus: [REQUIRED]
            The initial status of the Qualification type.
            Constraints: Valid values are: Active | Inactive
            

    :type RetryDelayInSeconds: integer
    :param RetryDelayInSeconds: The number of seconds that a Worker must wait after requesting a Qualification of the Qualification type before the worker can retry the Qualification request.
            Constraints: None. If not specified, retries are disabled and Workers can request a Qualification of this type only once, even if the Worker has not been granted the Qualification. It is not possible to disable retries for a Qualification type after it has been created with retries enabled. If you want to disable retries, you must delete existing retry-enabled Qualification type and then create a new Qualification type with retries disabled.
            

    :type Test: string
    :param Test: The questions for the Qualification test a Worker must answer correctly to obtain a Qualification of this type. If this parameter is specified, TestDurationInSeconds must also be specified.
            Constraints: Must not be longer than 65535 bytes. Must be a QuestionForm data structure. This parameter cannot be specified if AutoGranted is true.
            Constraints: None. If not specified, the Worker may request the Qualification without answering any questions.
            

    :type AnswerKey: string
    :param AnswerKey: The answers to the Qualification test specified in the Test parameter, in the form of an AnswerKey data structure.
            Constraints: Must not be longer than 65535 bytes.
            Constraints: None. If not specified, you must process Qualification requests manually.
            

    :type TestDurationInSeconds: integer
    :param TestDurationInSeconds: The number of seconds the Worker has to complete the Qualification test, starting from the time the Worker requests the Qualification.

    :type AutoGranted: boolean
    :param AutoGranted: Specifies whether requests for the Qualification type are granted immediately, without prompting the Worker with a Qualification test.
            Constraints: If the Test parameter is specified, this parameter cannot be true.
            

    :type AutoGrantedValue: integer
    :param AutoGrantedValue: The Qualification value to use for automatically granted Qualifications. This parameter is used only if the AutoGranted parameter is true.

    :rtype: dict
    :return: {
        'QualificationType': {
            'QualificationTypeId': 'string',
            'CreationTime': datetime(2015, 1, 1),
            'Name': 'string',
            'Description': 'string',
            'Keywords': 'string',
            'QualificationTypeStatus': 'Active'|'Inactive',
            'Test': 'string',
            'TestDurationInSeconds': 123,
            'AnswerKey': 'string',
            'RetryDelayInSeconds': 123,
            'IsRequestable': True|False,
            'AutoGranted': True|False,
            'AutoGrantedValue': 123
        }
    }
    
    
    """
    pass