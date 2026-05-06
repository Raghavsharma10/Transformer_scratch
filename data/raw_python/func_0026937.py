def security_code_date(self):
        """ Date of user's security code update """
        return sa.Column(
            sa.TIMESTAMP(timezone=False),
            default=datetime(2000, 1, 1),
            server_default="2000-01-01 01:01",
        )