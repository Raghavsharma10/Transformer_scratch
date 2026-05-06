def create_commerce():
        """
        Creates commerce from environment variables ``TBK_COMMERCE_ID``, ``TBK_COMMERCE_KEY``
        or for testing purposes ``TBK_COMMERCE_TESTING``.
        """
        commerce_id = os.getenv('TBK_COMMERCE_ID')
        commerce_key = os.getenv('TBK_COMMERCE_KEY')
        commerce_testing = os.getenv('TBK_COMMERCE_TESTING') == 'True'

        if not commerce_testing:
            if commerce_id is None:
                raise ValueError("create_commerce needs TBK_COMMERCE_ID environment variable")
            if commerce_key is None:
                raise ValueError("create_commerce needs TBK_COMMERCE_KEY environment variable")

        return Commerce(
            id=commerce_id or Commerce.TEST_COMMERCE_ID,
            key=commerce_key,
            testing=commerce_testing
        )