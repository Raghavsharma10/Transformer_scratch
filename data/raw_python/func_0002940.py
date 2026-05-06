def sendSMS(self, CorpNum, ItemCode, MgtKey, Sender, Receiver, Contents, UserID=None):
        """ 알림문자 전송
            args
                CorpNum : 팝빌회원 사업자번호
                ItemCode : 명세서 종류 코드
                    [121 - 거래명세서], [122 - 청구서], [123 - 견적서],
                    [124 - 발주서], [125 - 입금표], [126 - 영수증]
                MgtKey : 파트너 문서관리번호
                Sender : 발신번호
                Receiver : 수신번호
                Contents : 문자메시지 내용(최대 90Byte), 최대길이를 초과한경우 길이가 조정되어 전송됨
                UserID : 팝빌 회원아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """

        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if ItemCode == None or ItemCode == "":
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")

        postData = self._stringtify({
            "sender": Sender,
            "receiver": Receiver,
            "contents": Contents
        })

        return self._httppost('/Statement/' + str(ItemCode) + '/' + MgtKey, postData, CorpNum, UserID, "SMS")