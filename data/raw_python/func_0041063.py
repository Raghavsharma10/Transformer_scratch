def validate(cls, data):
        """Validate input data matches expected failure ``dict`` format."""
        try:
            jsonschema.validate(
                data, cls.SCHEMA,
                # See: https://github.com/Julian/jsonschema/issues/148
                types={'array': (list, tuple)})
        except jsonschema.ValidationError as e:
            raise InvalidFormat("Failure data not of the"
                                " expected format: %s" % (e.message))
        else:
            # Ensure that all 'exc_type_names' originate from one of
            # base exceptions, because those are the root exceptions that
            # python mandates/provides and anything else is invalid...
            causes = collections.deque([data])
            while causes:
                cause = causes.popleft()
                try:
                    generated_on = cause['generated_on']
                    ok_bases = cls.BASE_EXCEPTIONS[generated_on[0]]
                except (KeyError, IndexError):
                    ok_bases = []
                root_exc_type = cause['exc_type_names'][-1]
                if root_exc_type not in ok_bases:
                    raise InvalidFormat(
                        "Failure data 'exc_type_names' must"
                        " have an initial exception type that is one"
                        " of %s types: '%s' is not one of those"
                        " types" % (ok_bases, root_exc_type))
                sub_cause = cause.get('cause')
                if sub_cause is not None:
                    causes.append(sub_cause)