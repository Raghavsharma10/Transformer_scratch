def joinMember(self, JoinInfo):
        """ 팝빌 회원가입
            args
                JoinInfo : 회원가입정보. Reference JoinForm class
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        JoinInfo.LinkID = self.__linkID
        postData = self._stringtify(JoinInfo)
        return self._httppost('/Join', postData)