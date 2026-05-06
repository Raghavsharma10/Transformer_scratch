def change_password(self, new_password):
        """
        修改教务密码, **注意** 合肥校区使用信息中心账号登录, 与教务密码不一致, 即使修改了也没有作用, 因此合肥校区帐号调用此接口会直接报错

        @structure bool

        :param new_password: 新密码
        """

        if self.session.campus == HF:
            raise ValueError('合肥校区使用信息中心账号登录, 修改教务密码没有作用')
        # 若新密码与原密码相同, 直接返回 True
        if new_password == self.session.password:
            raise ValueError('原密码与新密码相同')

        result = self.query(ChangePassword(self.session.password, new_password))
        if result:
            self.session.password = new_password
        return result