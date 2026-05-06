def clean_ingested(self):
        """"Clean ingested files"""
        for s in self.sources:
            df = s.datafile
            if df.exists and not s.is_partition:
                df.remove()
                s.state = s.STATES.NEW

        self.commit()