def do_login(self, line):
        "login aws-acces-key aws-secret"
        if line:
            args = self.getargs(line)

            self.conn = boto.connect_dynamodb(
                aws_access_key_id=args[0],
                aws_secret_access_key=args[1])
        else:
            self.conn = boto.connect_dynamodb()

        self.do_tables('')