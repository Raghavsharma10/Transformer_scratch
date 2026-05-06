def list(self, request, *args, **kwargs):
        """
        Available request parameters:

        - ?type=type_of_statistics_objects (required. Have to be from the list: 'customer', 'project')
        - ?from=timestamp (default: now - 30 days, for example: 1415910025)
        - ?to=timestamp (default: now, for example: 1415912625)
        - ?datapoints=how many data points have to be in answer (default: 6)

        Answer will be list of datapoints(dictionaries).
        Each datapoint will contain fields: 'to', 'from', 'value'.
        'Value' - count of objects, that were created between 'from' and 'to' dates.

        Example:

        .. code-block:: javascript

            [
                {"to": 471970877, "from": 1, "value": 5},
                {"to": 943941753, "from": 471970877, "value": 0},
                {"to": 1415912629, "from": 943941753, "value": 3}
            ]
        """
        return super(CreationTimeStatsView, self).list(request, *args, **kwargs)