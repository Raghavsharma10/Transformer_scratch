def do_refresh(self,args):
        """Refresh the view of the eni"""
        pprint(AwsConnectionFactory.getEc2Client().describe_network_interfaces(NetworkInterfaceIds=[self.physicalId]));