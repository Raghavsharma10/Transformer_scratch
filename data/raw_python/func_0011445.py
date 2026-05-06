def revoke_token(self, subid):
    '''revoke a token by removing it. Is done at finish, and also available
    as a command line option'''
    from expfactory.database.models import Participant
    p = Participant.query.filter(Participant.id == subid).first()
    if p is not None:
        p.token = 'revoked'
    self.session.commit()
    return p