def search(self, CorpNum, JobID, Type, TaxType, PurposeType, TaxRegIDType, TaxRegIDYN, TaxRegID, Page, PerPage,
               Order, UserID=None):
        """ 수집 결과 조회
            args
                CorpNum : 팝빌회원 사업자번호
                JobID : 작업아이디
                Type : 문서형태 배열, N-일반전자세금계산서, M-수정전자세금계산서
                TaxType : 과세형태 배열, T-과세, N-면세, Z-영세
                PurposeType : 영수/청구, R-영수, C-청구, N-없음
                TaxRegIDType : 종사업장번호 사업자유형, S-공급자, B-공급받는자, T-수탁자
                TaxRegIDYN : 종사업장번호 유무, 공백-전체조회, 0-종사업장번호 없음, 1-종사업장번호 있음
                TaxRegID : 종사업장번호, 콤마(",")로 구분 하여 구성 ex) '0001,0002'
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

        uri = '/HomeTax/Taxinvoice/' + JobID
        uri += '?Type=' + ','.join(Type)
        uri += '&TaxType=' + ','.join(TaxType)
        uri += '&PurposeType=' + ','.join(PurposeType)
        uri += '&TaxRegIDType=' + TaxRegIDType
        uri += '&TaxRegID=' + TaxRegID
        uri += '&Page=' + str(Page)
        uri += '&PerPage=' + str(PerPage)
        uri += '&Order=' + Order

        if TaxRegIDYN != '':
            uri += '&TaxRegIDYN=' + TaxRegIDYN

        return self._httpget(uri, CorpNum, UserID)