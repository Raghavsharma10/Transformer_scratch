def delete(self, db_session=None):
        """
        Deletes the object via session, this will permanently delete the
        object from storage on commit

        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session, self)
        db_session.delete(self)