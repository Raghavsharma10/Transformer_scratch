def update(self, CorpNum, ItemCode, MgtKey, Statement, UserID=None):
        """ 임시저장
            args
                CorpNum : 팝빌회원 사업자번호
                ItemCode : 명세서 종류 코드
                    [121 - 거래명세서], [122 - 청구서], [123 - 견적서],
                    [124 - 발주서], [125 - 입금표], [126 - 영수증]
                MgtKey : 파트너 문서관리번호
                Statement : 등록할 전자명세서 object. made with Statement(...)
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """

        if Statement == None:
            raise PopbillException(-99999999, "등록할 전자명세서 정보가 입력되지 않았습니다.")
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if ItemCode == None or ItemCode == "":
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")

        postData = self._stringtify(Statement)

        return self._httppost('/Statement/' + str(ItemCode) + '/' + MgtKey, postData, CorpNum, UserID, 'PATCH')