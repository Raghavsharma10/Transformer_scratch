def validate_q_time(self, value):
        """
        Would be for example: [2013-03-01 TO 2013-04-01T00:00:00] and/or [* TO *]
        Returns a valid sorl value. [2013-03-01T00:00:00Z TO 2013-04-01T00:00:00Z] and/or [* TO *]
        """
        if value:
            try:
                range = utils.parse_datetime_range_to_solr(value)
                return range
            except Exception as e:
                raise serializers.ValidationError(e.message)

        return value