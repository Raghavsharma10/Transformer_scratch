def id(self):
        """ Unique identifier of user object"""
        return sa.Column(sa.Integer, primary_key=True, autoincrement=True)