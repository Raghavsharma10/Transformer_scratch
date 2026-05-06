def make_request_validator(request):
    """Validate arguments in incomming request."""
    verb = request.values.get('verb', '', type=str)
    resumption_token = request.values.get('resumptionToken', None)

    schema = Verbs if resumption_token is None else ResumptionVerbs
    return getattr(schema, verb, OAISchema)(partial=False)