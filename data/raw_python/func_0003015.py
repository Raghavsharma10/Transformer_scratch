def search(self, CorpNum, MgtKeyType, DType, SDate, EDate, State, Type, TaxType, LateOnly, TaxRegIDYN, TaxRegIDType,
               TaxRegID, Page, PerPage, Order, UserID=None, QString=None, InterOPYN=None, IssueType=None):
        """ 목록 조회
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKeyType : 세금계산서유형, SELL-매출, BUY-매입, TRUSTEE-위수탁
                DType : 일자유형, R-등록일시, W-작성일자, I-발행일시 중 택 1
                SDate : 시작일자, 표시형식(yyyyMMdd)
                EDate : 종료일자, 표시형식(yyyyMMdd)
                State : 상태코드, 2,3번째 자리에 와일드카드(*) 사용가능
                Type : 문서형태 배열, N-일반세금계산서, M-수정세금계산서
                TaxType : 과세형태 배열, T-과세, N-면세, Z-영세
                LateOnly : 지연발행, 공백-전체조회, 0-정상발행조회, 1-지연발행 조회
                TaxRegIdYN : 종사업장번호 유무, 공백-전체조회, 0-종사업장번호 없음 1-종사업장번호 있음
                TaxRegIDType : 종사업장번호 사업자유형, S-공급자, B-공급받는자, T-수탁자
                TaxRegID : 종사업장번호, 콤마(,)로 구분하여 구성 ex)'0001,1234'
                Page : 페이지번호
                PerPage : 페이지당 목록개수
                Order : 정렬방향, D-내림차순, A-오름차순
                UserID : 팝빌 회원아이디
                QString : 거래처 정보, 거래처 상호 또는 사업자등록번호 기재, 미기재시 전체조회
                InterOPYN : 연동문서 여부, 공백-전체조회, 0-일반문서 조회, 1-연동문서 조회
                IssueType : 발행형태 배열, N-정발행, R-역발행, T-위수탁
            return
                조회목록 Object
            raise
                PopbillException
        """

        if MgtKeyType not in self.__MgtKeyTypes:
            raise PopbillException(-99999999, "관리번호 형태가 올바르지 않습니다.")

        if DType == None or DType == '':
            raise PopbillException(-99999999, "일자유형이 입력되지 않았습니다.")

        if SDate == None or SDate == '':
            raise PopbillException(-99999999, "시작일자가 입력되지 않았습니다.")

        if EDate == None or EDate == '':
            raise PopbillException(-99999999, "종료일자가 입력되지 않았습니다.")

        uri = '/Taxinvoice/' + MgtKeyType
        uri += '?DType=' + DType
        uri += '&SDate=' + SDate
        uri += '&EDate=' + EDate
        uri += '&State=' + ','.join(State)
        uri += '&Type=' + ','.join(Type)
        uri += '&TaxType=' + ','.join(TaxType)
        uri += '&TaxRegIDType=' + TaxRegIDType
        uri += '&TaxRegID=' + TaxRegID
        uri += '&Page=' + str(Page)
        uri += '&PerPage=' + str(PerPage)
        uri += '&Order=' + Order
        uri += '&InterOPYN=' + InterOPYN

        if LateOnly != '':
            uri += '&LateOnly=' + LateOnly
        if TaxRegIDYN != '':
            uri += '&TaxRegIDType=' + TaxRegIDType

        if QString is not None:
            uri += '&QString=' + QString

        if IssueType is not None:
            uri += '&IssueType=' + ','.join(IssueType)

        return self._httpget(uri, CorpNum, UserID)