def users_for_perms(cls, perm_names, db_session=None):
        """
        return users hat have one of given permissions

        :param perm_names:
        :param db_session:
        :return:
        """
        db_session = get_db_session(db_session)
        query = db_session.query(cls.model)
        query = query.filter(
            cls.models_proxy.User.id == cls.models_proxy.UserGroup.user_id
        )
        query = query.filter(
            cls.models_proxy.UserGroup.group_id
            == cls.models_proxy.GroupPermission.group_id
        )
        query = query.filter(cls.models_proxy.GroupPermission.perm_name.in_(perm_names))

        query2 = db_session.query(cls.model)
        query2 = query2.filter(
            cls.models_proxy.User.id == cls.models_proxy.UserPermission.user_id
        )
        query2 = query2.filter(
            cls.models_proxy.UserPermission.perm_name.in_(perm_names)
        )
        users = query.union(query2).order_by(cls.model.id)
        return users