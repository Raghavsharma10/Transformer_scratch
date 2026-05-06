def virsh_version(self,
                      host_list=None,
                      remote_user=None,
                      remote_pass=None,
                      sudo=False,
                      sudo_user=None,
                      sudo_pass=None):
        '''
        Get the virsh version
        '''
        host_list, remote_user, remote_pass, \
            sudo, sudo_user, sudo_pass = self.get_validated_params(
               host_list, remote_user, remote_pass, sudo, sudo_user,
               sudo_pass)


        result, failed_hosts = self.runner.ansible_perform_operation(
            host_list=host_list,
            remote_user=remote_user,
            remote_pass=remote_pass,
            module="command",
            module_args="virsh version",
            sudo=sudo,
            sudo_user=sudo_user,
            sudo_pass=sudo_pass)

        virsh_result = None

        if result['contacted'].keys():
            virsh_result = {}
            for node in result['contacted'].keys():
                nodeobj = result['contacted'][node]
                jsonoutput = rex.parse_lrvalue_string(nodeobj['stdout'], ":")
                virsh_result[node] = {}
                virsh_result[node]['result'] = jsonoutput

        return virsh_result