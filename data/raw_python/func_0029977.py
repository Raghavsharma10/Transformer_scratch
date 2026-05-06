def clean_except_files(self):
        """Clean everything except the build source files"""

        if self.is_finalized:
            self.warn("Can't clean; bundle is finalized")
            return False

        self.log('---- Cleaning ----')
        self.state = self.STATES.CLEANING

        self.commit()

        self.clean_sources()
        self.clean_tables()
        self.clean_partitions()
        self.clean_build()
        self.clean_ingested()
        self.clean_build_state()

        self.state = self.STATES.CLEANED

        self.commit()

        self.log('---- Done Cleaning ----')

        return True