def common_install_apache2(self):
        """
            Install apache2 web server
        """
        try:
            sudo('apt-get install apache2 -y')
        except Exception as e:
            print(e)

        print(green(' * Installed Apache2 in the system.'))
        print(green(' * Done'))
        print()