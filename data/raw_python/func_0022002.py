def _check_response(cls, response, content_type=JSON_CONTENT_TYPE):
        """
        Check response content and its type.

        ..  note::

            Unlike :mod:`acme.client`, checking is strict.

        :param bytes content_type: Expected Content-Type response header.  If
            the response Content-Type does not match, :exc:`ClientError` is
            raised.

        :raises .ServerError: If server response body carries HTTP Problem
            (draft-ietf-appsawg-http-problem-00).
        :raises ~acme.errors.ClientError: In case of other networking errors.
        """
        def _got_failure(f):
            f.trap(ValueError)
            return None

        def _got_json(jobj):
            if 400 <= response.code < 600:
                if response_ct == JSON_ERROR_CONTENT_TYPE and jobj is not None:
                    raise ServerError(
                        messages.Error.from_json(jobj), response)
                else:
                    # response is not JSON object
                    raise errors.ClientError(response)
            elif response_ct != content_type:
                raise errors.ClientError(
                    'Unexpected response Content-Type: {0!r}'.format(
                        response_ct))
            elif content_type == JSON_CONTENT_TYPE and jobj is None:
                raise errors.ClientError(response)
            return response

        response_ct = response.headers.getRawHeaders(
            b'Content-Type', [None])[0]
        action = LOG_JWS_CHECK_RESPONSE(
            expected_content_type=content_type,
            response_content_type=response_ct)
        with action.context():
            # TODO: response.json() is called twice, once here, and
            # once in _get and _post clients
            return (
                DeferredContext(response.json())
                .addErrback(_got_failure)
                .addCallback(_got_json)
                .addActionFinish())