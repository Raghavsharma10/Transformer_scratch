def common_install_mysql(self):
        """
            Install mysql
        """
        sudo("debconf-set-selections <<< 'mysql-server mysql-server/root_password password {0}'".format(self.mysql_password))
        sudo("debconf-set-selections <<< 'mysql-server mysql-server/root_password_again password {0}'".format(self.mysql_password))
        sudo('apt-get install mysql-server -y')

        print(green(' * Installed MySql server in the system.'))
        print(green(' * Done'))
        print()