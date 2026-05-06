def updateEmailConfig(self, Corpnum, EmailType, SendYN, UserID=None):
        """ 알림메일 전송설정 수정
            args
                CorpNum : 팝빌회원 사업자번호
                EmailType: 메일전송유형
                SendYN: 전송여부 (True-전송, False-미전송)
                UserID : 팝빌회원 아이디
            return
               처리결과. consist of code and message
            raise
                PopbillException
        """
        if EmailType == None or EmailType == '':
            raise PopbillException(-99999999, "메일전송 타입이 입력되지 않았습니다.")

        if SendYN == None or SendYN == '':
            raise PopbillException(-99999999, "메일전송 여부 항목이 입력되지 않았습니다.")

        uri = "/Cashbill/EmailSendConfig?EmailType=" + EmailType + "&SendYN=" + str(SendYN)
        return self._httppost(uri, "", Corpnum, UserID)