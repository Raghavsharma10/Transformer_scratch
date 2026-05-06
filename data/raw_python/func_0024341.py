def dispatch_request(self, body):
        """Given a parsed JSON request object, call the correct Intent, Launch,
        or SessionEnded function.

        This function is called after request parsing and validaion and will
        raise a `ValueError` if an unknown request type comes in.

        :param body: JSON object loaded from incoming request's POST data.
        """

        req_type = body.get('request', {}).get('type')
        session_obj = body.get('session')

        session = Session(session_obj) if session_obj else None

        if req_type == 'LaunchRequest':
            return self.launch_fn(session)

        elif req_type == 'IntentRequest':
            intent = body['request']['intent']['name']
            intent_fn = self.intent_map.get(intent, self.unknown_intent_fn)

            slots = {
                slot['name']: slot.get('value')
                for _, slot in
                body['request']['intent'].get('slots', {}).items()
            }

            arity = intent_fn.__code__.co_argcount

            if arity == 2:
                return intent_fn(slots, session)

            return intent_fn()

        elif req_type == 'SessionEndedRequest':
            return self.session_end_fn()

        log.error('invalid request type: %s', req_type)
        raise ValueError('bad request: %s', body)