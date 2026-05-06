def requestJob(self, CorpNum, Type, SDate, EDate, UserID=None):
        """ 수집 요청
            args
                CorpNum : 팝빌회원 사업자번호
                Type : 문서형태, SELL-매출, BUY-매입,
                SDate : 시작일자, 표시형식(yyyyMMdd)
                EDate : 종료일자, 표시형식(yyyyMMdd)
                UserID : 팝빌회원 아이디
            return
                작업아이디 (jobID)
            raise
                PopbillException
        """

        if Type == None or Type == '':
            raise PopbillException(-99999999, "문서형태이 입력되지 않았습니다.")

        if SDate == None or SDate == '':
            raise PopbillException(-99999999, "시작일자가 입력되지 않았습니다.")

        if EDate == None or EDate == '':
            raise PopbillException(-99999999, "종료일자가 입력되지 않았습니다.")

        uri = '/HomeTax/Cashbill/' + Type
        uri += '?SDate=' + SDate
        uri += '&EDate=' + EDate

        return self._httppost(uri, "", CorpNum, UserID).jobID