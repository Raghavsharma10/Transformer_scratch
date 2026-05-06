def get_vpc_id(self):
        """Gets the VPC ID for this EC2 instance

        :return: String instance ID or None
        """
        log = logging.getLogger(self.cls_logger + '.get_vpc_id')

        # Exit if not running on AWS
        if not self.is_aws:
            log.info('This machine is not running in AWS, exiting...')
            return

        if self.instance_id is None:
            log.error('Unable to get the Instance ID for this machine')
            return
        log.info('Found Instance ID: {i}'.format(i=self.instance_id))

        log.info('Querying AWS to get the VPC ID...')
        try:
            response = self.client.describe_instances(
                    DryRun=False,
                    InstanceIds=[self.instance_id])
        except ClientError as ex:
            log.error('Unable to query AWS to get info for instance {i}\n{e}'.format(
                    i=self.instance_id, e=ex))
            return

        # Get the VPC ID from the response
        try:
            vpc_id = response['Reservations'][0]['Instances'][0]['VpcId']
        except KeyError as ex:
            log.error('Unable to get VPC ID from response: {r}\n{e}'.format(r=response, e=ex))
            return
        log.info('Found VPC ID: {v}'.format(v=vpc_id))
        return vpc_id