def audit_1_3(self):
        """1.3 Ensure credentials unused for 90 days or greater are disabled (Scored)"""
        for row in self.credential_report:
            for access_key in "1", "2":
                if json.loads(row["access_key_{}_active".format(access_key)]):
                    last_used = row["access_key_{}_last_used_date".format(access_key)]
                    if last_used != "N/A" and self.parse_date(last_used) < datetime.now(tzutc()) - timedelta(days=90):
                        msg = "Active access key {} in account {} last used over 90 days ago"
                        raise Exception(msg.format(access_key, row["user"]))