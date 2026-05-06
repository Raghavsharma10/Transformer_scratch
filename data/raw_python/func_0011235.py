def parse_environment(fields, context, topics):
    """Resolve the be.yaml environment key

    Features:
        - Lists, e.g. ["/path1", "/path2"]
        - Environment variable references, via $
        - Replacement field references, e.g. {key}
        - Topic references, e.g. {1}

    """

    def _resolve_environment_lists(context):
        """Concatenate environment lists"""
        for key, value in context.copy().iteritems():
            if isinstance(value, list):
                context[key] = os.pathsep.join(value)
        return context

    def _resolve_environment_references(fields, context):
        """Resolve $ occurences by expansion

        Given a dictionary {"PATH": "$PATH;somevalue;{0}"}
        Return {"PATH": "value_of_PATH;somevalue;myproject"},
        given that the first topic - {0} - is "myproject"

        Arguments:
            fields (dict): Environment from be.yaml
            context (dict): Source context

        """

        def repl(match):
            key = pattern[match.start():match.end()].strip("$")
            return context.get(key)

        pat = re.compile("\$\w+", re.IGNORECASE)
        for key, pattern in fields.copy().iteritems():
            fields[key] = pat.sub(repl, pattern) \
                          .strip(os.pathsep)  # Remove superflous separators

        return fields

    def _resolve_environment_fields(fields, context, topics):
        """Resolve {} occurences

        Supports both positional and BE_-prefixed variables.

        Example:
            BE_MYKEY -> "{mykey}" from `BE_MYKEY`
            {1} -> "{mytask}" from `be in myproject mytask`

        Returns:
            Dictionary of resolved fields

        """

        source_dict = replacement_fields_from_context(context)
        source_dict.update(dict((str(topics.index(topic)), topic)
                                for topic in topics))

        def repl(match):
            key = pattern[match.start():match.end()].strip("{}")
            try:
                return source_dict[key]
            except KeyError:
                echo("PROJECT ERROR: Unavailable reference \"%s\" "
                     "in be.yaml" % key)
                sys.exit(PROJECT_ERROR)

        for key, pattern in fields.copy().iteritems():
            fields[key] = re.sub("{[\d\w]+}", repl, pattern)

        return fields

    fields = _resolve_environment_lists(fields)
    fields = _resolve_environment_references(fields, context)
    fields = _resolve_environment_fields(fields, context, topics)

    return fields