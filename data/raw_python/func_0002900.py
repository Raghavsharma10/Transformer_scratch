def getEPrintURL(self, CorpNum, MgtKey, UserID=None):
        """ 공급받는자용 인쇄 URL 확인
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKey : 문서관리번호
                UserID : 팝빌회원 아이디
            return
                팝빌 URL as str
            raise
                PopbillException
        """
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        result = self._httpget('/Cashbill/' + MgtKey + '?TG=EPRINT', CorpNum, UserID)

        return result.url