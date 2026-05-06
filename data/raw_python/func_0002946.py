def attachStatement(self, CorpNum, ItemCode, MgtKey, SubItemCode, SubMgtKey, UserID=None):
        """ 다른 전자명세서 첨부
            args
                CorpNum : 팝빌회원 사업자번호
                ItemCode : 전자명세서 종류코드, 121-명세서, 122-청구서, 123-견적서, 124-발주서 125-입금표, 126-영수증
                MgtKey : 전자명세서 문서관리번호
                SubItemCode : 첨부할 명세서 종류코드, 121-명세서, 122-청구서, 123-견적서, 124-발주서 125-입금표, 126-영수증
                SubMgtKey : 첨부할 전자명세서 문서관리번호
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if ItemCode == None or ItemCode == "":
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")

        uri = '/Statement/' + ItemCode + '/' + MgtKey + '/AttachStmt'

        postData = self._stringtify({"ItemCode": ItemCode, "MgtKey": SubMgtKey})

        return self._httppost(uri, postData, CorpNum, UserID)