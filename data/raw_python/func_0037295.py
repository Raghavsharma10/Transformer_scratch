def do_account_info(self):
        """display account information"""
        s, metadata = self.n.getRegisterUserInfo()
        pprint.PrettyPrinter(indent=2).pprint(metadata)