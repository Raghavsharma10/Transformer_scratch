def dispatch(self, request, **kwargs):
        '''
        Entry point for this class, here we decide basic stuff
        '''

        # Delete method must happen with POST not with GET
        if request.method == 'POST':
            # Check if this is a webservice request
            self.__authtoken = (bool(getattr(self.request, "authtoken", False)))
            self.json_worker = self.__authtoken or (self.json is True)
            # Call the base implementation
            return super(GenDelete, self).dispatch(request, **kwargs)
        else:
            json_answer = json.dumps({
                'error': True,
                'errortxt': _('Method not allowed, use POST to delete or DELETE on the detail url'),
                })
            return HttpResponse(json_answer, content_type='application/json')