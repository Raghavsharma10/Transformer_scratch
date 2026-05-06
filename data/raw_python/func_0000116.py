def return_values(self):
        """ Guess what api we are using and return as public api does.
        Private has {'id':'key', 'value':'keyvalue'} format, public has {'key':'keyvalue'}
        """

        j = self.json()
        #TODO: FIXME: get rid of old API when its support will be removed
        public_api_value = j.get('returnValues')
        old_private_value = j.get('endpoints')
        new_private_value = self.__collect_interfaces_return(j.get('interfaces', {}))

        retvals = new_private_value or old_private_value or public_api_value or []
        # TODO: Public api hack.
        if self._router.public_api_in_use:
            return retvals
        return self.__parse(retvals)