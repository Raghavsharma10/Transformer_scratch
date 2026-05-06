def sendEmail(self, CorpNum, MgtKey, ReceiverEmail, UserID=None):
        """ 알림메일 재전송
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKey : 문서관리번호
                ReceiverEmail : 수신자 이메일 주소
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """

        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if ReceiverEmail == None or ReceiverEmail == "":
            raise PopbillException(-99999999, "수신자 메일주소가 입력되지 않았습니다.")

        postData = self._stringtify({"receiver": ReceiverEmail})

        return self._httppost('/Cashbill/' + MgtKey, postData, CorpNum, UserID, "EMAIL")