def create_v4_signature(self, request_params):
        '''
        Create URI and signature headers based on AWS V4 signing process.
        Refer to https://docs.aws.amazon.com/AlexaWebInfoService/latest/ApiReferenceArticle.html for request params.
        :param request_params: dictionary of request parameters
        :return: URL and header to be passed to requests.get
        '''

        method = 'GET'
        service = 'awis'
        host = 'awis.us-west-1.amazonaws.com'
        region = 'us-west-1'
        endpoint = 'https://awis.amazonaws.com/api'
        request_parameters = urlencode([(key, request_params[key]) for key in sorted(request_params.keys())])

        # Key derivation functions. See:
        # http://docs.aws.amazon.com/general/latest/gr/signature-v4-examples.html#signature-v4-examples-python
        def sign(key, msg):
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

        def getSignatureKey(key, dateStamp, regionName, serviceName):
            kDate = sign(('AWS4' + key).encode('utf-8'), dateStamp)
            kRegion = sign(kDate, regionName)
            kService = sign(kRegion, serviceName)
            kSigning = sign(kService, 'aws4_request')
            return kSigning

        # Create a date for headers and the credential string
        t = datetime.datetime.utcnow()
        amzdate = t.strftime('%Y%m%dT%H%M%SZ')
        datestamp = t.strftime('%Y%m%d') # Date w/o time, used in credential scope

        # Create canonical request
        canonical_uri = '/api'
        canonical_querystring = request_parameters
        canonical_headers = 'host:' + host + '\n' + 'x-amz-date:' + amzdate + '\n'
        signed_headers = 'host;x-amz-date'
        payload_hash = hashlib.sha256(''.encode('utf8')).hexdigest()
        canonical_request = method + '\n' + canonical_uri + '\n' + canonical_querystring + '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash

        # Create string to sign
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = datestamp + '/' + region + '/' + service + '/' + 'aws4_request'
        string_to_sign = algorithm + '\n' +  amzdate + '\n' +  credential_scope + '\n' +  hashlib.sha256(canonical_request.encode('utf8')).hexdigest()

        # Calculate signature
        signing_key = getSignatureKey(self.secret_access_key, datestamp, region, service)

        # Sign the string_to_sign using the signing_key
        signature = hmac.new(signing_key, (string_to_sign).encode('utf-8'), hashlib.sha256).hexdigest()

        # Add signing information to the request
        authorization_header = algorithm + ' ' + 'Credential=' + self.access_id + '/' + credential_scope + ', ' +  'SignedHeaders=' + signed_headers + ', ' + 'Signature=' + signature
        headers = {'X-Amz-Date':amzdate, 'Authorization':authorization_header, 'Content-Type': 'application/xml', 'Accept': 'application/xml'}

        # Create request url
        request_url = endpoint + '?' + canonical_querystring

        return request_url, headers