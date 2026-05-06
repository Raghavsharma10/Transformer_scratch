def generate(self, verified, keygen):
        """
        :param verified: телефон или email (verified_entity)
        :param keygen: функция генерации ключа
        :return:
        """
        return Verification(
            verified_entity=verified,
            key=keygen(),
            verified=False
        )