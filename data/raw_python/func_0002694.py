def getSentListURL(self, CorpNum, UserID):
        """
        카카오톡 전송내역 팝업 URL
        :param CorpNum: 팝빌회원 사업자번호
        :param UserID: 팝빌회원 아이디
        :return: 팝빌 URL
        """
        result = self._httpget('/KakaoTalk/?TG=BOX', CorpNum, UserID)
        return result.url