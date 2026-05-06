def schedule_ping_frequency(self):  # pragma: no cover
        "Send a ping message to slack every 20 seconds"
        ping = crontab('* * * * * */20', func=self.send_ping, start=False)
        ping.start()