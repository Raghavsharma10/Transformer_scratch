def _do_api_call(self, call_type=u'', id = None):
        """
        returns a response if it is a valid call
        otherwise the corresponding error
        
        """
        endpoint = None
        url = None
        # print "call_type %s" % (call_type)
        if call_type == u'load':
            endpoint = self.get_endpoint("self")
            if id is None:
                raise APIException("LOAD_IDNOTSET", "could not load object")
        elif call_type == u'delete' or (call_type == u"destroy"):
            endpoint = self.get_endpoint("destroy")
            if id is None:
                raise APIException("DELETE_IDNOTSET", "could not delete object")
        elif call_type == u'update':
            endpoint = self.get_endpoint("update")
            if id is None:
                raise APIException("UPDATE_IDNOTSET", "could not load object")
        elif call_type == u'create':
            endpoint = self.get_endpoint("create")
            url = u"%s%s%s" % (self.__api__.base_url, API_BASE_PATH, endpoint['href'])
            # post?
        elif call_type == u'schema':
            # add schema gethering functionality 
            # hackisch
            endpoint = self.get_endpoint("create")
            url = u"%s%s%s/schema" % (self.__api__.base_url, API_BASE_PATH, endpoint['href'])
            endpoint['method'] = u'GET'    
        if id is not None:
            url = u"%s%s%s" % (self.__api__.base_url, API_BASE_PATH, endpoint['href'].replace(u"{id}",id))
        ## excecute the api request
        payload = self.to_json()
        if u'method' in endpoint.keys():
            method = endpoint['method']
        else:
            method = u'GET'
        # request raises exceptions if something goes wrong
        obj = None
        try:
            # dbg
            msg = u"url: %s method:%s p: %s" % (url, method, payload)
            #print msg
            response = self.__api__.request(url, method, data=payload)
            #load update create success
            if ((response.status_code == 200 and 
                 call_type in ['load', 'update']) or
            (response.status_code == 201 and call_type == 'create')):
                msg = "call_type: %s successfully completed" % call_type
                log.info(msg)
                return self.to_instance(response)
            elif (response.status_code == 200 and call_type in ['delete', 'destroy']):
            #delete success
                msg ="call_type: %s successfully completed" % call_type
                log.info(msg)
                return self._try_to_serialize(response)
            elif 200 <= response.status_code <= 299:
                return self._try_to_serialize(response)
        except Exception as e:
            msg = "Exception occoured %s url: %s" % (e, url)
            log.error(msg)
            raise e