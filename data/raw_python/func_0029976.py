def clean(self, force=False):
        """Clean generated objects from the dataset, but only if there are File contents
         to regenerate them"""

        if self.is_finalized and not force:
            self.warn("Can't clean; bundle is finalized")
            return False

        self.log('---- Cleaning ----')
        self.state = self.STATES.CLEANING
        self.dstate = self.STATES.BUILDING

        self.commit()

        self.clean_sources()
        self.clean_tables()
        self.clean_partitions()
        self.clean_build()
        self.clean_files()
        self.clean_ingested()
        self.clean_build_state()
        self.clean_progress()

        self.state = self.STATES.CLEANED

        self.commit()

        return True