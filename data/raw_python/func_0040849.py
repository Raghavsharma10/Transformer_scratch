def generate_ssh(self, server, args, configure):
        """
        异步同时执行SSH生成 generate ssh
        :param server:
        :param args:
        :param configure:
        :return:
        """
        self.reset_server_env(server, configure)

        # chmod project root owner
        sudo('chown {user}:{user} -R {path}'.format(
            user=configure[server]['user'],
            path=bigdata_conf.project_root
        ))

        # generate ssh key
        if not exists('~/.ssh/id_rsa.pub'):
            run('ssh-keygen -t rsa -P "" -f ~/.ssh/id_rsa')