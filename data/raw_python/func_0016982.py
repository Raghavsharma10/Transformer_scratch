def audit_1_12(self):
        """1.12 Ensure no root account access key exists (Scored)"""
        for row in self.credential_report:
            if row["user"] == "<root_account>":
                self.assertFalse(json.loads(row["access_key_1_active"]))
                self.assertFalse(json.loads(row["access_key_2_active"]))