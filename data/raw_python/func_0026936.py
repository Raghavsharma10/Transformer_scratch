def last_login_date(self):
        """ Date of user's last login """
        return sa.Column(
            sa.TIMESTAMP(timezone=False),
            default=lambda x: datetime.utcnow(),
            server_default=sa.func.now(),
        )