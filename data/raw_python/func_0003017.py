def assignMgtKey(self, CorpNum, MgtKeyType, ItemKey, MgtKey, UserID=None):
        """ 관리번호할당
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKeyType : 세금계산서 유형, SELL-매출, BUY-매입, TRUSTEE-위수탁
                ItemKey : 아이템키 (Search API로 조회 가능)
                MgtKey : 세금계산서에 할당할 파트너 관리 번호
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if MgtKeyType == None or MgtKeyType == '':
            raise PopbillException(-99999999, "세금계산서 발행유형이 입력되지 않았습니다.")

        if ItemKey == None or ItemKey == '':
            raise PopbillException(-99999999, "아이템키가 입력되지 않았습니다.")

        if MgtKey == None or MgtKey == '':
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        postDate = "MgtKey=" + MgtKey
        return self._httppost('/Taxinvoice/' + ItemKey + '/' + MgtKeyType, postDate, CorpNum, UserID, "",
                              "application/x-www-form-urlencoded; charset=utf-8")