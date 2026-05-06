def getURL(self, CorpNum, UserID, ToGo):
        """
        :param CorpNum: 팝빌회원 사업자번호
        :param UserID: 팝빌회원 아이디
        :param ToGo: [PLUSFRIEND-플러스친구계정관리, SENDER-발신번호관리, TEMPLATE-알림톡템플릿관리, BOX-카카오톡전송내용]
        :return: 팝빌 URL
        """
        if ToGo == None or ToGo == '':
            raise PopbillException(-99999999, "TOGO값이 입력되지 않았습니다.")

        if ToGo == 'SENDER':
            result = self._httpget('/Message/?TG=' + ToGo, CorpNum, UserID)
        else:
            result = self._httpget('/KakaoTalk/?TG=' + ToGo, CorpNum, UserID)
        return result.url