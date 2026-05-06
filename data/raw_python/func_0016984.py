def audit_1_15(self):
        """1.15 Ensure IAM policies are attached only to groups or roles (Scored)"""
        for policy in resources.iam.policies.all():
            self.assertEqual(len(list(policy.attached_users.all())), 0, "{} has users attached to it".format(policy))