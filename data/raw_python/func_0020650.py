def parse_home_face_offs(self):
        """
        Parse only the home faceoffs
        
        :returns: ``self`` on success, ``None`` otherwise
        """
        self.__set_team_docs()
        self.face_offs['home'] = FaceOffRep.__read_team_doc(self.__home_doc)
        return self