def build_payload(cls, method, args):
        """ Build the payload to be sent to a `Responder` """
        ref = str(uuid.uuid4())
        return (method, args, ref)