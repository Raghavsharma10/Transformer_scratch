def history(self):
        """Returns an SQLAlchemy query of the object's history (previous
        versions). If the class does not support history/versioning,
        returns None.
        """
        history = self.history_class
        if history:
            return self.session.query(history).filter(history.id == self.id)
        else:
            return None