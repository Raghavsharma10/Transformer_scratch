def cmd_work(self, connection, sender, target, payload):
        """
        Does some job
        """
        connection.action(target, "is doing something...")
        time.sleep(int(payload or "5"))
        connection.action(target, "has finished !")
        connection.privmsg(target, "My answer is: 42.")