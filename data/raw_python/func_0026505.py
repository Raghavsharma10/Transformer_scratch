def check_alerts(self):
        """Periodical check to issue due alerts"""

        alerted = []

        for alert_time, task in self.alerts.items():
            task_time = dateutil.parser.parse(alert_time)
            if task_time < get_time():
                self.log('Alerting about task now:', task)

                address = objectmodels['user'].find_one({'uuid': task.owner}).mail
                subject = "Task alert: %s" % task.name
                text = """Task alert is due:\n%s""" % task.notes

                self.fireEvent(send_mail(address, subject, text))

                alerted.append(task.alert_time)

        for item in alerted:
            del self.alerts[item]