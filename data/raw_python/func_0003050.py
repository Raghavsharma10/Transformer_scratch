def updateCorpInfo(self, CorpNum, CorpInfo, UserID=None):
        """ 담당자 정보 수정
            args
                CorpNum : 회원 사업자번호
                CorpInfo : 회사 정보, Reference CorpInfo class
                UserID :  회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        postData = self._stringtify(CorpInfo)
        return self._httppost('/CorpInfo', postData, CorpNum, UserID)