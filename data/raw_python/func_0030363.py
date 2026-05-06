def execute_tropo_program(self, program):
        """
        Ask Tropo to execute a program for us.

        We can't do this directly;
        we have to ask Tropo to call us back and then give Tropo the
        program in the response body to that request from Tropo.

        But we can pass data to Tropo and ask Tropo to pass it back
        to us when Tropo calls us back. So, we just bundle up the program
        and pass it to Tropo, then when Tropo calls us back, we
        give the program back to Tropo.

        We also cryptographically sign our program, so that
        we can verify when we're called back with a program, that it's
        one that we sent to Tropo and has not gotten mangled.

        See https://docs.djangoproject.com/en/1.4/topics/signing/ for more
        about the signing API.

        See https://www.tropo.com/docs/webapi/passing_in_parameters_text.htm
        for the format we're using to call Tropo, pass it data, and ask
        them to call us back.



        :param program: A Tropo program, i.e. a dictionary with a 'tropo'
            key whose value is a list of dictionaries, each representing
            a Tropo command.
        """
        # The signer will also "pickle" the data structure for us
        signed_program = signing.dumps(program)

        params = {
            'action': 'create',  # Required by Tropo
            'token': self.config['messaging_token'],  # Identify ourselves
            'program': signed_program,  # Additional data
        }
        data = json.dumps(params)

        # Tell Tropo we'd like our response in JSON format
        # and our data is in that format too.
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        response = requests.post(base_url,
                                 data=data,
                                 headers=headers)

        # If the HTTP request failed, raise an appropriate exception - e.g.
        # if our network (or Tropo) are down:
        response.raise_for_status()

        result = json.loads(response.content)
        if not result['success']:
            raise Exception("Tropo error: %s" % result.get('error', 'unknown'))