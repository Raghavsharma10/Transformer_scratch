def shift_ordering_up(cls, parent_id, position, db_session=None, *args, **kwargs):
        """
        Shifts ordering to "open a gap" for node insertion,
        begins the shift from given position

        :param parent_id:
        :param position:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model)
        query = query.filter(cls.model.parent_id == parent_id)
        query = query.filter(cls.model.ordering >= position)
        query.update(
            {cls.model.ordering: cls.model.ordering + 1}, synchronize_session=False
        )
        db_session.flush()