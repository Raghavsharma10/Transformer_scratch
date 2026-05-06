def getPlusFriendMgtURL(self, CorpNum, UserID):
        """
        플러스친구 계정관리 팝업 URL
        :param CorpNum: 팝빌회원 사업자번호
        :param UserID: 팝빌회원 아이디
        :return: 팝빌 URL
        """
        result = self._httpget('/KakaoTalk/?TG=PLUSFRIEND', CorpNum, UserID)
        return result.url