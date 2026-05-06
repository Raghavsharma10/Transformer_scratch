def revokeRegistIssue(self, CorpNum, mgtKey, orgConfirmNum, orgTradeDate, smssendYN=False, memo=None, UserID=None,
                          isPartCancel=False, cancelType=None, supplyCost=None, tax=None, serviceFee=None,
                          totalAmount=None):
        """ 취소현금영수증 즉시발행
            args
                CorpNum : 팝빌회원 사업자번호
                mgtKey : 현금영수증 문서관리번호
                orgConfirmNum : 원본현금영수증 승인번호
                orgTradeDate : 원본현금영수증 거래일자
                smssendYN : 발행안내문자 전송여부
                memo : 메모
                UserID : 팝빌회원 아이디
                isPartCancel : 부분취소여부
                cancelType : 취소사유
                supplyCost : [취소] 공급가액
                tax : [취소] 세액
                serviceFee : [취소] 봉사료
                totalAmount : [취소] 합계금액
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """

        postData = self._stringtify({
            "mgtKey": mgtKey,
            "orgConfirmNum": orgConfirmNum,
            "orgTradeDate": orgTradeDate,
            "smssendYN": smssendYN,
            "memo": memo,
            "isPartCancel": isPartCancel,
            "cancelType": cancelType,
            "supplyCost": supplyCost,
            "tax": tax,
            "serviceFee": serviceFee,
            "totalAmount": totalAmount,
        })

        return self._httppost('/Cashbill', postData, CorpNum, UserID, "REVOKEISSUE")