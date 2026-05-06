def validate_token(self, token):
    '''retrieve a subject based on a token. Valid means we return a participant
       invalid means we return None
    '''
    from expfactory.database.models import Participant
    p = Participant.query.filter(Participant.token == token).first()
    if p is not None:
        if p.token.endswith(('finished','revoked')):
            p = None
        else:
            p = p.id
    return p