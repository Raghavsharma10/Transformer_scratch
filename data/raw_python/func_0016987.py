def audit_2_4(self):
        """2.4 Ensure CloudTrail trails are integrated with CloudWatch Logs (Scored)"""
        for trail in self.trails:
            self.assertIn("CloudWatchLogsLogGroupArn", trail)
            trail_status = clients.cloudtrail.get_trail_status(Name=trail["TrailARN"])
            self.assertGreater(trail_status["LatestCloudWatchLogsDeliveryTime"],
                               datetime.now(tzutc()) - timedelta(days=1))