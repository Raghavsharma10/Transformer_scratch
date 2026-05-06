def set_telephone(self, tel):
        """
        更新电话

        @structure bool

        :param tel: 电话号码, 需要满足手机和普通电话的格式, 例如 `18112345678` 或者 '0791-1234567'
        """

        return type(tel)(self.query(SetTelephone(tel))) == tel