def generate_subid(self, token=None, return_user=False):
    '''generate a new user in the database, still session based so we
       create a new identifier.
    '''    
    from expfactory.database.models import Participant
    if not token:
        p = Participant()
    else:
        p = Participant(token=token)
    self.session.add(p)
    self.session.commit()
    if return_user is True:
        return p
    return p.id