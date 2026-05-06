def search(self, CorpNum, DType, SDate, EDate, State, ItemCode, Page, PerPage, Order, UserID=None, QString=None):
        """ 목록 조회
            args
                CorpNum : 팝빌회원 사업자번호
                DType : 일자유형, R-등록일시, W-작성일자, I-발행일시 중 택 1
                SDate : 시작일자, 표시형식(yyyyMMdd)
                EDate : 종료일자, 표시형식(yyyyMMdd)
                State : 상태코드, 2,3번째 자리에 와일드카드(*) 사용가능
                ItemCode : 명세서 종류코드 배열, 121-명세서, 122-청구서, 123-견적서, 124-발주서 125-입금표, 126-영수증
                Page : 페이지번호
                PerPage : 페이지당 목록개수
                Order : 정렬방향, D-내림차순, A-오름차순
                QString : 거래처 정보, 거래처 상호 또는 사업자등록번호 기재, 미기재시 전체조회
                UserID : 팝빌 회원아이디
        """

        if DType == None or DType == '':
            raise PopbillException(-99999999, "일자유형이 입력되지 않았습니다.")

        if SDate == None or SDate == '':
            raise PopbillException(-99999999, "시작일자가 입력되지 않았습니다.")

        if EDate == None or EDate == '':
            raise PopbillException(-99999999, "종료일자가 입력되지 않았습니다.")

        uri = '/Statement/Search'
        uri += '?DType=' + DType
        uri += '&SDate=' + SDate
        uri += '&EDate=' + EDate
        uri += '&State=' + ','.join(State)
        uri += '&ItemCode=' + ','.join(ItemCode)
        uri += '&Page=' + str(Page)
        uri += '&PerPage=' + str(PerPage)
        uri += '&Order=' + Order

        if QString is not None:
            uri += '&QString=' + QString

        return self._httpget(uri, CorpNum, UserID)