def search(self, CorpNum, JobID, TradeType, TradeUsage, Page, PerPage, Order, UserID=None):
        """ 수집 결과 조회
            args
                CorpNum : 팝빌회원 사업자번호
                JobID : 작업아이디
                TradeType : 문서형태 배열, N-일반 현금영수증, C-취소 현금영수증
                TradeUsage : 거래구분 배열, P-소등공제용, C-지출증빙용
                Page : 페이지 번호
                PerPage : 페이지당 목록 개수, 최대 1000개
                Order : 정렬 방향, D-내림차순, A-오름차순
                UserID : 팝빌회원 아이디
            return
                수집 결과 정보
            raise
                PopbillException
        """
        if JobID == None or len(JobID) != 18:
            raise PopbillException(-99999999, "작업아이디(jobID)가 올바르지 않습니다.")

        uri = '/HomeTax/Cashbill/' + JobID
        uri += '?TradeType=' + ','.join(TradeType)
        uri += '&TradeUsage=' + ','.join(TradeUsage)
        uri += '&Page=' + str(Page)
        uri += '&PerPage=' + str(PerPage)
        uri += '&Order=' + Order

        return self._httpget(uri, CorpNum, UserID)