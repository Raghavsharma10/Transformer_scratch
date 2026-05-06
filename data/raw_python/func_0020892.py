def validation_error(exception):
    """Return formatter validation error."""
    messages = getattr(exception, 'messages', None)
    if messages is None:
        messages = getattr(exception, 'data', {'messages': None})['messages']

    def extract_errors():
        """Extract errors from exception."""
        if isinstance(messages, dict):
            for field, message in messages.items():
                if field == 'verb':
                    yield 'badVerb', '\n'.join(message)
                else:
                    yield 'badArgument', '\n'.join(message)
        else:
            for field in exception.field_names:
                if field == 'verb':
                    yield 'badVerb', '\n'.join(messages)
                else:
                    yield 'badArgument', '\n'.join(messages)

            if not exception.field_names:
                yield 'badArgument', '\n'.join(messages)

    return (etree.tostring(xml.error(extract_errors())),
            422,
            {'Content-Type': 'text/xml'})