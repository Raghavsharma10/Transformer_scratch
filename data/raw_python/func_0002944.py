def deleteFile(self, CorpNum, ItemCode, MgtKey, FileID, UserID=None):
        """ 첨부파일 삭제
            args
                CorpNum : 팝빌회원 사업자번호
                ItemCode : 명세서 종류 코드
                    [121 - 거래명세서], [122 - 청구서], [123 - 견적서],
                    [124 - 발주서], [125 - 입금표], [126 - 영수증]
                MgtKey : 파트너 문서관리번호
                FileID : 파일아이디, 첨부파일 목록확인(getFiles) API 응답전문의 AttachedFile 변수값
                UserID : 팝빌회원 아이디
            return
                첨부파일 정보 목록 as List
            raise
                PopbillException
        """
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if ItemCode == None or ItemCode == "":
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")
        if FileID == None or FileID == "":
            raise PopbillException(-99999999, "파일아이디가 입력되지 않았습니다.")

        postData = ''

        return self._httppost('/Statement/' + str(ItemCode) + '/' + MgtKey + '/Files/' + FileID, postData, CorpNum,
                              UserID, 'DELETE')