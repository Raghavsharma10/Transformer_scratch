def attachStatement(self, CorpNum, MgtKeyType, MgtKey, ItemCode, StmtMgtKey, UserID=None):
        """ 전자명세서 첨부
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKeyType : 세금계산서 유형, SELL-매출, BUY-매입, TRUSTEE-위수탁
                MgtKey : 세금계산서 문서관리번호
                StmtCode : 명세서 종류코드, 121-명세서, 122-청구서, 123-견적서, 124-발주서 125-입금표, 126-영수증
                StmtMgtKey : 전자명세서 문서관리번호
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """

        if MgtKeyType not in self.__MgtKeyTypes:
            raise PopbillException(-99999999, "관리번호 형태가 올바르지 않습니다.")
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        uri = '/Taxinvoice/' + MgtKeyType + '/' + MgtKey + '/AttachStmt'

        postData = self._stringtify({"ItemCode": ItemCode, "MgtKey": StmtMgtKey})

        return self._httppost(uri, postData, CorpNum, UserID)