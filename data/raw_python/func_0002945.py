def getMassPrintURL(self, CorpNum, ItemCode, MgtKeyList, UserID=None):
        """ 다량 인쇄 URL 확인
            args
                CorpNum : 팝빌회원 사업자번호
                ItemCode : 명세서 종류 코드
                    [121 - 거래명세서], [122 - 청구서], [123 - 견적서],
                    [124 - 발주서], [125 - 입금표], [126 - 영수증]
                MgtKeyList : 파트너 문서관리번호 목록
                UserID : 팝빌회원 아이디
            return
                팝빌 URL as str
            raise
                PopbillException
        """
        if MgtKeyList == None:
            raise PopbillException(-99999999, "관리번호 배열이 입력되지 않았습니다.")
        if ItemCode == None or ItemCode == "":
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")

        postData = self._stringtify(MgtKeyList)

        result = self._httppost('/Statement/' + str(ItemCode) + '?Print', postData, CorpNum, UserID)

        return result.url