def json_serial(obj):
        """
        Custom JSON serializer for objects not serializable by default.
        """

        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()

        raise TypeError('Type {} not serializable.'.format(type(obj)))