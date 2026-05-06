def save(self, commit=False):
        """Save the changes to the model.

        If the model has not been persisted
        then it adds the model to the declared session. Then it flushes the
        object session and optionally commits it.
        """
        if not has_identity(self):
            # Object has not been persisted to the database.
            session.add(self)

        if commit:
            # Commit the session as requested.
            session.commit()

        else:
            # Just flush the session; do not commit.
            session.flush()