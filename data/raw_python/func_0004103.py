def main(self):
        """Run the command
        """
        self._init_config()

        if self.dry_run:
            return self.run_dry_run()
        elif self.watch:
            return self.run_watch()
        else:
            return self.run_render()