def audit_2_5(self):
        """2.5 Ensure AWS Config is enabled in all regions (Scored)"""
        import boto3
        for region in boto3.Session().get_available_regions("config"):
            aws_config = boto3.session.Session(region_name=region).client("config")
            res = aws_config.describe_configuration_recorder_status()
            self.assertGreater(len(res["ConfigurationRecordersStatus"]), 0)