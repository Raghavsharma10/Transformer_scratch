def audit_1_1(self):
        """1.1 Avoid the use of the "root" account (Scored)"""
        for row in self.credential_report:
            if row["user"] == "<root_account>":
                for field in "password_last_used", "access_key_1_last_used_date", "access_key_2_last_used_date":
                    if row[field] != "N/A" and self.parse_date(row[field]) > datetime.now(tzutc()) - timedelta(days=1):
                        raise Exception("Root account last used less than a day ago ({})".format(field))