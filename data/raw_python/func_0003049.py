def updateContact(self, CorpNum, ContactInfo, UserID=None):
        """ 담당자 정보 수정
            args
                CorpNum : 회원 사업자번호
                ContactInfo : 담당자 정보, Reference ContactInfo class
                UserID :  회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        postData = self._stringtify(ContactInfo)
        return self._httppost('/IDs', postData, CorpNum, UserID)