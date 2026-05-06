def shift_ordering_down(
        self, parent_id, position, db_session=None, *args, **kwargs
    ):
        """
        Shifts ordering to "close gaps" after node deletion or being moved
        to another branch, begins the shift from given position

        :param parent_id:
        :param position:
        :param db_session:
        :return:
        """
        return self.service.shift_ordering_down(
            parent_id=parent_id,
            position=position,
            db_session=db_session,
            *args,
            **kwargs
        )