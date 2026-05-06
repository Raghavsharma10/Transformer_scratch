def search(self, CorpNum, DType, SDate, EDate, State, TradeType, TradeUsage, TaxationType, Page, PerPage, Order,
               UserID=None, QString=None, TradeOpt=None):
        """ 목록 조회
            args
                CorpNum : 팝빌회원 사업자번호
                DType : 일자유형, R-등록일자, T-거래일자, I-발행일자 중 택 1
                SDate : 시작일자, 표시형식(yyyyMMdd)
                EDate : 종료일자, 표시형식(yyyyMMdd)
                State : 상태코드 배열, 2,3번째 자리에 와일드카드(*) 사용가능
                TradeType : 문서형태 배열, N-일반현금영수증, C-취소현금영수증
                TradeUsage : 거래구분 배열, P-소득공제용, C-지출증빙용
                TaxationType : 과세형태 배열, T-과세, N-비과세
                Page : 페이지번호
                PerPage : 페이지당 검색개수
                Order : 정렬방향, D-내림차순, A-오름차순
                UserID : 팝빌 회원아이디
                QString : 현금영수증 식별번호, 미기재시 전체조회
                TradeOpt : 거래유형, N-일반, B-도서공연, T-대중교통
        """

        if DType == None or DType == '':
            raise PopbillException(-99999999, "일자유형이 입력되지 않았습니다.")

        if SDate == None or SDate == '':
            raise PopbillException(-99999999, "시작일자가 입력되지 않았습니다.")

        if EDate == None or EDate == '':
            raise PopbillException(-99999999, "종료일자가 입력되지 않았습니다.")

        uri = '/Cashbill/Search'
        uri += '?DType=' + DType
        uri += '&SDate=' + SDate
        uri += '&EDate=' + EDate
        uri += '&State=' + ','.join(State)
        uri += '&TradeUsage=' + ','.join(TradeUsage)
        uri += '&TradeType=' + ','.join(TradeType)
        uri += '&TaxationType=' + ','.join(TaxationType)
        uri += '&Page=' + str(Page)
        uri += '&PerPage=' + str(PerPage)
        uri += '&Order=' + Order

        if QString is not None:
            uri += '&QString=' + QString

        if TradeOpt is not None:
            uri += '&TradeOpt=' + ','.join(TradeOpt)

        return self._httpget(uri, CorpNum, UserID)