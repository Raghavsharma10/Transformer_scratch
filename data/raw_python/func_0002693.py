def getATSTemplateMgtURL(self, CorpNum, UserID):
        """
        알림톡 템플릿관리 팝업 URL
        :param CorpNum: 팝빌회원 사업자번호
        :param UserID: 팝빌회원 아이디
        :return: 팝빌 URL
        """
        result = self._httpget('/KakaoTalk/?TG=TEMPLATE', CorpNum, UserID)
        return result.url