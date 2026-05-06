def issue(self, CorpNum, ItemCode, MgtKey, Memo=None, EmailSubject=None, UserID=None):
        """ 발행
            args
                CorpNum : 팝빌회원 사업자번호
                ItemCode : 명세서 종류 코드
                    [121 - 거래명세서], [122 - 청구서], [123 - 견적서],
                    [124 - 발주서], [125 - 입금표], [126 - 영수증]
                MgtKey : 파트너 문서관리번호
                Memo : 처리메모
                EmailSubject : 발행메일 제목(미기재시 기본양식으로 전송)
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if ItemCode == None or ItemCode == "":
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")

        req = {}
        postData = ""

        if Memo != None and Memo != '':
            req["memo"] = Memo
        if EmailSubject != None and EmailSubject != '':
            req["emailSubject"] = EmailSubject

        postData = self._stringtify(req)

        return self._httppost('/Statement/' + str(ItemCode) + '/' + MgtKey, postData, CorpNum, UserID, "ISSUE")