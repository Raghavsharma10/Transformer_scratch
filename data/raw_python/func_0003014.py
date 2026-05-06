def getMassPrintURL(self, CorpNum, MgtKeyType, MgtKeyList, UserID=None):
        """ 다량 인쇄 URL 확인
            args
                CorpNum : 회원 사업자 번호
                MgtKeyType : 관리번호 유형 one of ['SELL','BUY','TRUSTEE']
                MgtKeyList : 파트너 관리번호 목록
                UserID : 팝빌 회원아이디
            return
                팝빌 URL as str
            raise
                PopbillException
        """
        if MgtKeyList == None or len(MgtKeyList) < 1:
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        postData = self._stringtify(MgtKeyList)

        Result = self._httppost('/Taxinvoice/' + MgtKeyType + "?Print", postData, CorpNum, UserID)

        return Result.url