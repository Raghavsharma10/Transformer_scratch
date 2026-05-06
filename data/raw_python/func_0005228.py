def verify(verified_entity, verification_key):
        """
        Метод должен райзить ошибки
        :param verified_entity: сущность
        :param verification_key: ключ
        :return:
        """
        verification = get_object_or_none(Verification, verified_entity=verified_entity)

        if verification is None:
            raise ServerError(VerificationHandler.STATUS_VERIFICATION_NOT_FOUND)
        if not verification.verify(verification_key):
            raise ServerError(VerificationHandler.STATUS_INVALID_VERIFICATION_KEY)

        verification.verified = True
        verification.save()