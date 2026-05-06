def common_update_sys(self):
        """
            update system package
        """
        try:
            sudo('apt-get update -y --fix-missing')
        except Exception as e:
            print(e)

        print(green('System package is up to date.'))
        print()