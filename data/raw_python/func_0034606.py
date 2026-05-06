def connect(self, region):
        ''' create connection to api server'''
        if self.eucalyptus:
            conn = boto.connect_euca(host=self.eucalyptus_host)
            conn.APIVersion = '2010-08-31'
        else:
            conn = self.connect_to_aws(ec2, region)
        return conn