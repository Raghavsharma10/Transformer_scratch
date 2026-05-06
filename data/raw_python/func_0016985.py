def audit_2_2(self):
        """2.2 Ensure CloudTrail log file validation is enabled (Scored)"""
        self.assertGreater(len(self.trails), 0, "No CloudTrail trails configured")
        self.assertTrue(all(trail["LogFileValidationEnabled"] for trail in self.trails),
                        "Some CloudTrail trails don't have log file validation enabled")