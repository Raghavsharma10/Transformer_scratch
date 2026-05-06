def rados_df(self,
                 host_list=None,
                 remote_user=None,
                 remote_pass=None):
        '''
        Invoked the rados df command and return output to user
        '''
        result, failed_hosts = self.runner.ansible_perform_operation(
            host_list=host_list,
            remote_user=remote_user,
            remote_pass=remote_pass,
            module="command",
            module_args="rados df")

        parsed_result = self.rados_parse_df(result)

        return parsed_result