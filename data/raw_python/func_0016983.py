def audit_1_13(self):
        """1.13 Ensure hardware MFA is enabled for the "root" account (Scored)"""
        for row in self.credential_report:
            if row["user"] == "<root_account>":
                self.assertTrue(json.loads(row["mfa_active"]))