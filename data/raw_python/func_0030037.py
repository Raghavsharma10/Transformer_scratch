def spec(self):
        """Return a SourceSpec to describe this source"""
        from ambry_sources.sources import SourceSpec

        d = self.dict
        d['url'] = self.url

        # Will get the URL twice; once as ref and once as URL, but the ref is ignored

        return SourceSpec(**d)