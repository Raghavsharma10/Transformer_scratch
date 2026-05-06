def parse(self):
        """
        Retreive and parse Play by Play data for the given nhlscrapi.GameKey
        
        :returns: ``self`` on success, ``None`` otherwise
        """
        try:
            return (
                super(FaceOffRep, self).parse()
                and self.parse_home_face_offs()
                and self.parse_away_face_offs()
            )
        except:
            return None