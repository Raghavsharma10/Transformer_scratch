def _read_from_hdx(self, object_type, value, fieldname='id',
                       action=None, **kwargs):
        # type: (str, str, str, Optional[str], Any) -> Tuple[bool, Union[Dict, str]]
        """Makes a read call to HDX passing in given parameter.

        Args:
            object_type (str): Description of HDX object type (for messages)
            value (str): Value of HDX field
            fieldname (str): HDX field name. Defaults to id.
            action (Optional[str]): Replacement CKAN action url to use. Defaults to None.
            **kwargs: Other fields to pass to CKAN.

        Returns:
            Tuple[bool, Union[Dict, str]]: (True/False, HDX object metadata/Error)
        """
        if not fieldname:
            raise HDXError('Empty %s field name!' % object_type)
        if action is None:
            action = self.actions()['show']
        data = {fieldname: value}
        data.update(kwargs)
        try:
            result = self.configuration.call_remoteckan(action, data)
            return True, result
        except NotFound:
            return False, '%s=%s: not found!' % (fieldname, value)
        except Exception as e:
            raisefrom(HDXError, 'Failed when trying to read: %s=%s! (POST)' % (fieldname, value), e)