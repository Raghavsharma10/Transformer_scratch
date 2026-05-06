def validate_a_time_filter(self, value):
        """
        Would be for example: [2013-03-01 TO 2013-04-01:00:00:00] and/or [* TO *]
        """
        if value:
            try:
                utils.parse_datetime_range(value)
            except Exception as e:
                raise serializers.ValidationError(e.message)

        return value