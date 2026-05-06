def do_tail(self,args):
        """Tail the logs"""
        response = AwsConnectionFactory.getLogClient().get_log_events(
                logGroupName=self.logStream['logGroupName'],
                logStreamName=self.logStream['logStreamName'],
                limit=10,
                startFromHead=False
            )
        pprint(response)