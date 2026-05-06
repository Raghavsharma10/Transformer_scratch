def list_users(self, user=None):
    '''list users, each having a model in the database. A headless experiment
       will use protected tokens, and interactive will be based on auto-
       incremented ids.
    ''' 
    from expfactory.database.models import Participant
    participants = Participant.query.all()
    users = []
    for user in participants:
        users.append(self.print_user(user))
    return users