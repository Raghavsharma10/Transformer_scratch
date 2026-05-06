def shift_ordering_up(self, parent_id, position, db_session=None, *args, **kwargs):
        """
        Shifts ordering to "open a gap" for node insertion,
        begins the shift from given position

        :param parent_id:
        :param position:
        :param db_session:
        :return:
        """
        return self.service.shift_ordering_up(
            parent_id=parent_id,
            position=position,
            db_session=db_session,
            *args,
            **kwargs
        )