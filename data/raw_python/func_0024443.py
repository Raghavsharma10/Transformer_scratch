def sendoff_current_user(self):
        """
        Tell current user that s/he finished it's job for now.
        We'll notify if workflow arrives again to his/her WF Lane.
        """
        msgs = self.task_data.get('LANE_CHANGE_MSG', DEFAULT_LANE_CHANGE_MSG)
        self.msg_box(title=msgs['title'], msg=msgs['body'])