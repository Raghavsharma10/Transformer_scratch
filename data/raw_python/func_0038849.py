def log(self, msg):
        """
        Log a message information adding the master_class and instance_class if available

        :param msg:
        :return:
        """
        if self.master_class and self.instance_class:
            logger.info('{0} - {1} - {2} - {3} - lang: {4} msg: {5}'.format(
                self.ct_master.app_label, self.ct_master.model,
                self.instance_class, self.instance.language_code, self.instance.pk, msg)
            )
        elif self.instance_class:
            logger.info('{} - {}: {}'.format(self.instance_class, self.instance.pk, msg))
        else:
            logger.info('{}'.format(msg))