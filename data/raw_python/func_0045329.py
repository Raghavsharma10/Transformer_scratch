def validate(self, request, data):
        """
        Validate response from OpenID server.
        Set identity in case of successfull validation.
        """
        client = consumer.Consumer(request.session, None)

        try:
            resp = client.complete(data, request.session['openid_return_to'])
        except KeyError:
            messages.error(request, lang.INVALID_RESPONSE_FROM_OPENID)
            return redirect('netauth-login')
        if resp.status == consumer.CANCEL:
            messages.warning(request, lang.OPENID_CANCELED)
            return redirect('netauth-login')
        elif resp.status == consumer.FAILURE:
            messages.error(request, lang.OPENID_FAILED % resp.message)
            return redirect('netauth-login')
        elif resp.status == consumer.SUCCESS:
            self.identity = resp.identity_url
            del request.session['openid_return_to']
            return resp