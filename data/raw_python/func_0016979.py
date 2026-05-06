def audit_1_2(self):
        """1.2 Ensure multi-factor authentication (MFA) is enabled for all IAM users that have a console password (Scored)"""  # noqa
        for row in self.credential_report:
            if row["user"] == "<root_account>" or json.loads(row["password_enabled"]):
                if not json.loads(row["mfa_active"]):
                    raise Exception("Account {} has a console password but no MFA".format(row["user"]))