def refresh_token(self, subid):
    '''refresh or generate a new token for a user'''
    from expfactory.database.models import Participant
    p = Participant.query.filter(Participant.id == subid).first()
    if p is not None:
        p.token = str(uuid.uuid4())
    self.session.commit()
    return p