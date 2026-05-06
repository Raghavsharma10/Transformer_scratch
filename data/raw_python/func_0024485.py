def wsgi_app(self, request):
        """Incoming request handler.

        :param request: Werkzeug request object
        """

        try:
            if request.method != 'POST':
                abort(400)

            try:
                # Python 2.7 compatibility
                data = request.data
                if isinstance(data, str):
                    body = json.loads(data)
                else:
                    body = json.loads(data.decode('utf-8'))
            except ValueError:
                abort(400)

            if self.validate:
                valid_cert = util.validate_request_certificate(
                    request.headers, request.data)

                valid_ts = util.validate_request_timestamp(body)

                if not valid_cert or not valid_ts:
                    log.error('failed to validate request')
                    abort(403)

            resp_obj = self.alexa.dispatch_request(body)
            return Response(response=json.dumps(resp_obj, indent=4),
                            status=200,
                            mimetype='application/json')

        except HTTPException as exc:
            log.exception('Failed to handle request')
            return exc