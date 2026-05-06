def send_cons3rt_agent_logs(self):
        """Sends a Slack message with an attachment for each cons3rt agent log

        :return:
        """
        log = logging.getLogger(self.cls_logger + '.send_cons3rt_agent_logs')

        log.debug('Searching for log files in directory: {d}'.format(d=self.dep.cons3rt_agent_log_dir))
        for item in os.listdir(self.dep.cons3rt_agent_log_dir):
            item_path = os.path.join(self.dep.cons3rt_agent_log_dir, item)
            if os.path.isfile(item_path):
                log.debug('Adding slack attachment with cons3rt agent log file: {f}'.format(f=item_path))
                try:
                    with open(item_path, 'r') as f:
                        file_text = f.read()
                except (IOError, OSError) as e:
                    log.warn('There was a problem opening file: {f}\n{e}'.format(f=item_path, e=e))
                    continue

                # Take the last 7000 characters
                file_text_trimmed = file_text[-7000:]
                attachment = SlackAttachment(fallback=file_text_trimmed, text=file_text_trimmed, color='#9400D3')
                self.add_attachment(attachment)
        self.send()