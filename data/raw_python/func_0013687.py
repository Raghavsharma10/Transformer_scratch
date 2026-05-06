def describe_batch_predictions(FilterVariable=None, EQ=None, GT=None, LT=None, GE=None, LE=None, NE=None, Prefix=None, SortOrder=None, NextToken=None, Limit=None):
    """
    Returns a list of BatchPrediction operations that match the search criteria in the request.
    See also: AWS API Documentation
    
    
    :example: response = client.describe_batch_predictions(
        FilterVariable='CreatedAt'|'LastUpdatedAt'|'Status'|'Name'|'IAMUser'|'MLModelId'|'DataSourceId'|'DataURI',
        EQ='string',
        GT='string',
        LT='string',
        GE='string',
        LE='string',
        NE='string',
        Prefix='string',
        SortOrder='asc'|'dsc',
        NextToken='string',
        Limit=123
    )
    
    
    :type FilterVariable: string
    :param FilterVariable: Use one of the following variables to filter a list of BatchPrediction :
            CreatedAt - Sets the search criteria to the BatchPrediction creation date.
            Status - Sets the search criteria to the BatchPrediction status.
            Name - Sets the search criteria to the contents of the BatchPrediction **** Name .
            IAMUser - Sets the search criteria to the user account that invoked the BatchPrediction creation.
            MLModelId - Sets the search criteria to the MLModel used in the BatchPrediction .
            DataSourceId - Sets the search criteria to the DataSource used in the BatchPrediction .
            DataURI - Sets the search criteria to the data file(s) used in the BatchPrediction . The URL can identify either a file or an Amazon Simple Storage Solution (Amazon S3) bucket or directory.
            

    :type EQ: string
    :param EQ: The equal to operator. The BatchPrediction results will have FilterVariable values that exactly match the value specified with EQ .

    :type GT: string
    :param GT: The greater than operator. The BatchPrediction results will have FilterVariable values that are greater than the value specified with GT .

    :type LT: string
    :param LT: The less than operator. The BatchPrediction results will have FilterVariable values that are less than the value specified with LT .

    :type GE: string
    :param GE: The greater than or equal to operator. The BatchPrediction results will have FilterVariable values that are greater than or equal to the value specified with GE .

    :type LE: string
    :param LE: The less than or equal to operator. The BatchPrediction results will have FilterVariable values that are less than or equal to the value specified with LE .

    :type NE: string
    :param NE: The not equal to operator. The BatchPrediction results will have FilterVariable values not equal to the value specified with NE .

    :type Prefix: string
    :param Prefix: A string that is found at the beginning of a variable, such as Name or Id .
            For example, a Batch Prediction operation could have the Name 2014-09-09-HolidayGiftMailer . To search for this BatchPrediction , select Name for the FilterVariable and any of the following strings for the Prefix :
            2014-09
            2014-09-09
            2014-09-09-Holiday
            

    :type SortOrder: string
    :param SortOrder: A two-value parameter that determines the sequence of the resulting list of MLModel s.
            asc - Arranges the list in ascending order (A-Z, 0-9).
            dsc - Arranges the list in descending order (Z-A, 9-0).
            Results are sorted by FilterVariable .
            

    :type NextToken: string
    :param NextToken: An ID of the page in the paginated results.

    :type Limit: integer
    :param Limit: The number of pages of information to include in the result. The range of acceptable values is 1 through 100 . The default value is 100 .

    :rtype: dict
    :return: {
        'Results': [
            {
                'BatchPredictionId': 'string',
                'MLModelId': 'string',
                'BatchPredictionDataSourceId': 'string',
                'InputDataLocationS3': 'string',
                'CreatedByIamUser': 'string',
                'CreatedAt': datetime(2015, 1, 1),
                'LastUpdatedAt': datetime(2015, 1, 1),
                'Name': 'string',
                'Status': 'PENDING'|'INPROGRESS'|'FAILED'|'COMPLETED'|'DELETED',
                'OutputUri': 'string',
                'Message': 'string',
                'ComputeTime': 123,
                'FinishedAt': datetime(2015, 1, 1),
                'StartedAt': datetime(2015, 1, 1),
                'TotalRecordCount': 123,
                'InvalidRecordCount': 123
            },
        ],
        'NextToken': 'string'
    }
    
    
    :returns: 
    PENDING - Amazon Machine Learning (Amazon ML) submitted a request to generate predictions for a batch of observations.
    INPROGRESS - The process is underway.
    FAILED - The request to perform a batch prediction did not run to completion. It is not usable.
    COMPLETED - The batch prediction process completed successfully.
    DELETED - The BatchPrediction is marked as deleted. It is not usable.
    
    """
    pass