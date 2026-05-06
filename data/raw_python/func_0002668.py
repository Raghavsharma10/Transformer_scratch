def summary(self, CorpNum, JobID, TradeType, TradeUsage, UserID=None):
        """ 수집 결과 요약정보 조회
            args
                CorpNum : 팝빌회원 사업자번호
                JobID : 작업아이디
                TradeType : 문서형태 배열, N-일반 현금영수증, C-취소 현금영수증
                TradeUsage : 거래구분 배열, P-소등공제용, C-지출증빙용
                UserID : 팝빌회원 아이디
            return
                수집 결과 요약정보
            raise
                PopbillException
        """
        if JobID == None or len(JobID) != 18:
            raise PopbillException(-99999999, "작업아이디(jobID)가 올바르지 않습니다.")

        uri = '/HomeTax/Cashbill/' + JobID + '/Summary'
        uri += '?TradeType=' + ','.join(TradeType)
        uri += '&TradeUsage=' + ','.join(TradeUsage)

        return self._httpget(uri, CorpNum, UserID)