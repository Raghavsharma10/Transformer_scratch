def nginx_web_ssl_config(self):
        """
           Nginx web ssl config
        """

        dt = [self.nginx_web_dir, self.nginx_ssl_dir]
        return nginx_conf_string.simple_ssl_web_conf.format(dt=dt)