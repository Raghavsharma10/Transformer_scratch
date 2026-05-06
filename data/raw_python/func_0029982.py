def clean_process_meta(self):
        """Remove all process and build metadata"""
        ds = self.dataset
        ds.config.build.clean()
        ds.config.process.clean()
        ds.commit()
        self.state = self.STATES.CLEANED