def _format_vpc_config(self):
        '''
        Returns {} if the VPC config is set to None by Config,
        returns the formatted config otherwise
        '''
        if self._config.raw['vpc']:
            return {
                'SubnetIds': self._config.raw['vpc']['subnets'],
                'SecurityGroupIds': self._config.raw['vpc']['security_groups']
            }
        else:
            return {
                'SubnetIds': [],
                'SecurityGroupIds': [],
            }