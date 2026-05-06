def common_install_nginx(self):
        """
            Install nginx
        """
        run('echo "deb http://ppa.launchpad.net/nginx/stable/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/nginx-stable.list')
        sudo('apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C300EE8C')
        sudo('apt-get update -y')
        sudo('apt-get install nginx -y')

        print(green(' * Installed Nginx in the system.'))
        print(green(' * Done'))
        print()