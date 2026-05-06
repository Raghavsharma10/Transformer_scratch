def on_status(self, status):
        """Print out some tweets"""
        self.out.write(json.dumps(status))
        self.out.write(os.linesep)

        self.received += 1
        return not self.terminate