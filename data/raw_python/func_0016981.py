def audit_1_4(self):
        """1.4 Ensure access keys are rotated every 90 days or less (Scored)"""
        for row in self.credential_report:
            for access_key in "1", "2":
                if json.loads(row["access_key_{}_active".format(access_key)]):
                    last_rotated = row["access_key_{}_last_rotated".format(access_key)]
                    if self.parse_date(last_rotated) < datetime.now(tzutc()) - timedelta(days=90):
                        msg = "Active access key {} in account {} last rotated over 90 days ago"
                        raise Exception(msg.format(access_key, row["user"]))