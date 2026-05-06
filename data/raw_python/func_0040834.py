def common_config_nginx_ssl(self):
        """
            Convert nginx server from http to https
        """
        if prompt(red(' * Change url from http to https (y/n)?'), default='n') == 'y':
            if not exists(self.nginx_ssl_dir):
                sudo('mkdir -p {0}'.format(self.nginx_ssl_dir))

            # generate ssh key
            sudo('openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout {0}/cert.key -out {0}/cert.pem'.format(self.nginx_ssl_dir))

            # do nginx config config
            put(StringIO(self.nginx_web_ssl_config), '/etc/nginx/sites-available/default', use_sudo=True)

            sudo('service nginx restart')

            print(green(' * Make Nginx from http to https.'))
            print(green(' * Done'))
            print()