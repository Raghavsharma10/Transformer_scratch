def terminate(self, arg):
        """
        Terminate instance with given EC2 ID or nametag.
        """
        instance = self.get(arg)
        with self.msg("Terminating %s (%s): " % (instance.name, instance.id)):
            instance.rename("old-%s" % instance.name)
            instance.terminate()
            while instance.state != 'terminated':
                time.sleep(5)
                self.log(".", end='')
                instance.update()