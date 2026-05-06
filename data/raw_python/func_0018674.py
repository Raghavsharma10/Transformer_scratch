def _build_time_fields_spec(name_in_context: str) -> List[Dict[str, Any]]:
        """
        Constructs the spec for predefined fields that are to be included in the master spec prior to schema load
        :param name_in_context: Name of the current object in the context
        :return:
        """
        return [
            {
                'Name': '_start_time',
                'Type': Type.DATETIME,
                'Value': ('time if {aggregate}._start_time is None else time '
                          'if time < {aggregate}._start_time else {aggregate}._start_time'
                          ).format(aggregate=name_in_context),
                ATTRIBUTE_INTERNAL: True
            },
            {
                'Name': '_end_time',
                'Type': Type.DATETIME,
                'Value': ('time if {aggregate}._end_time is None else time '
                          'if time > {aggregate}._end_time else {aggregate}._end_time'
                          ).format(aggregate=name_in_context),
                ATTRIBUTE_INTERNAL: True
            },
        ]