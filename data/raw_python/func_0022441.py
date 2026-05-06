def open_notification(self):
        """
        Open notification

        Built in support for Android 4.3 (API level 18)

        Using swipe action as a workaround for API level lower than 18

        """
        sdk_version = self.device.info['sdkInt']
        if sdk_version < 18:
            height = self.device.info['displayHeight']
            self.device.swipe(1, 1, 1, height - 1, 1)
        else:
            self.device.open.notification()