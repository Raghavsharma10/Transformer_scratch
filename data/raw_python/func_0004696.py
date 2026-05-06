def view_func(self):
        """Default view function for Flask app.

        This is a simple implementation for view func, you can add it to
        your Flask app::

            weixin = Weixin(app)
            app.add_url_rule('/', view_func=weixin.view_func)
        """
        if request is None:
            raise RuntimeError('view_func need Flask be installed')

        signature = request.args.get('signature')
        timestamp = request.args.get('timestamp')
        nonce = request.args.get('nonce')
        if not self.validate(signature, timestamp, nonce):
            return 'signature failed', 400

        if request.method == 'GET':
            echostr = request.args.get('echostr', '')
            return echostr

        try:
            ret = self.parse(request.data)
        except ValueError:
            return 'invalid', 400

        if 'type' not in ret:
            # not a valid message
            return 'invalid', 400

        if ret['type'] == 'text' and ret['content'] in self._registry:
            func = self._registry[ret['content']]
        else:
            ret_set = frozenset(ret.items())
            matched_rules = (
                _func for _func, _limitation in self._registry_without_key
                if _limitation.issubset(ret_set))
            func = next(matched_rules, None)  # first matched rule

        if func is None:
            if '*' in self._registry:
                func = self._registry['*']
            else:
                func = 'failed'

        if callable(func):
            text = func(**ret)
        else:
            # plain text
            text = self.reply(
                username=ret['sender'],
                sender=ret['receiver'],
                content=func,
            )

        return Response(text, content_type='text/xml; charset=utf-8')