def validate_q_geo(self, value):
        """
        Would be for example: [-90,-180 TO 90,180]
        """
        if value:
            try:
                rectangle = utils.parse_geo_box(value)
                return "[{0},{1} TO {2},{3}]".format(
                    rectangle.bounds[0],
                    rectangle.bounds[1],
                    rectangle.bounds[2],
                    rectangle.bounds[3],
                )
            except Exception as e:
                raise serializers.ValidationError(e.message)

        return value