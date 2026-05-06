def set_attributes(self, obj, **attributes):
        """ Set attributes.

        :param obj: requested object.
        :param attributes: dictionary of {attribute: value} to set
        """

        attributes_url = '{}/{}/attributes'.format(self.session_url, obj.ref)
        attributes_list = [{u'name': str(name), u'value': str(value)} for name, value in attributes.items()]
        self._request(RestMethod.patch, attributes_url, headers={'Content-Type': 'application/json'},
                      data=json.dumps(attributes_list))