def FAXSend(self, CorpNum, statement, SendNum, ReceiveNum, UserID=None):
        """ 선팩스 전송
            args
                CorpNum : 팝빌회원 사업자번호
                statement : 전자명세서 객체
                SendNum : 팩스 발신번호
                ReceiveNum : 팩스 수신번호
                UserID : 팝빌회원 아이디
            return
                팩스전송 접수번호(receiptNum)
            raise
                PopbillException
        """
        if statement == None:
            raise PopbillException(-99999999, "전송할 전자명세서 정보가 입력되지 않았습니다.")
        if SendNum == None or SendNum == '':
            raise PopbillException(-99999999, "팩스전송 발신번호가 올바르지 않았습니다.")
        if ReceiveNum == None or ReceiveNum == '':
            raise PopbillException(-99999999, "팩스전송 수신번호가 올바르지 않습니다.")

        statement.sendNum = SendNum
        statement.receiveNum = ReceiveNum

        postData = self._stringtify(statement)

        return self._httppost('/Statement', postData, CorpNum, UserID, "FAX").receiptNum