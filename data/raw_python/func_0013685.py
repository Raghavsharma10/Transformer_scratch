def put_bot(name=None, description=None, intents=None, clarificationPrompt=None, abortStatement=None, idleSessionTTLInSeconds=None, voiceId=None, checksum=None, processBehavior=None, locale=None, childDirected=None):
    """
    Creates an Amazon Lex conversational bot or replaces an existing bot. When you create or update a bot you only required to specify a name. You can use this to add intents later, or to remove intents from an existing bot. When you create a bot with a name only, the bot is created or updated but Amazon Lex returns the ```` response FAILED . You can build the bot after you add one or more intents. For more information about Amazon Lex bots, see  how-it-works .
    If you specify the name of an existing bot, the fields in the request replace the existing values in the $LATEST version of the bot. Amazon Lex removes any fields that you don't provide values for in the request, except for the idleTTLInSeconds and privacySettings fields, which are set to their default values. If you don't specify values for required fields, Amazon Lex throws an exception.
    This operation requires permissions for the lex:PutBot action. For more information, see  auth-and-access-control .
    See also: AWS API Documentation
    
    
    :example: response = client.put_bot(
        name='string',
        description='string',
        intents=[
            {
                'intentName': 'string',
                'intentVersion': 'string'
            },
        ],
        clarificationPrompt={
            'messages': [
                {
                    'contentType': 'PlainText'|'SSML',
                    'content': 'string'
                },
            ],
            'maxAttempts': 123,
            'responseCard': 'string'
        },
        abortStatement={
            'messages': [
                {
                    'contentType': 'PlainText'|'SSML',
                    'content': 'string'
                },
            ],
            'responseCard': 'string'
        },
        idleSessionTTLInSeconds=123,
        voiceId='string',
        checksum='string',
        processBehavior='SAVE'|'BUILD',
        locale='en-US',
        childDirected=True|False
    )
    
    
    :type name: string
    :param name: [REQUIRED]
            The name of the bot. The name is not case sensitive.
            

    :type description: string
    :param description: A description of the bot.

    :type intents: list
    :param intents: An array of Intent objects. Each intent represents a command that a user can express. For example, a pizza ordering bot might support an OrderPizza intent. For more information, see how-it-works .
            (dict) --Identifies the specific version of an intent.
            intentName (string) -- [REQUIRED]The name of the intent.
            intentVersion (string) -- [REQUIRED]The version of the intent.
            
            

    :type clarificationPrompt: dict
    :param clarificationPrompt: When Amazon Lex doesn't understand the user's intent, it uses one of these messages to get clarification. For example, 'Sorry, I didn't understand. Please repeat.' Amazon Lex repeats the clarification prompt the number of times specified in maxAttempts . If Amazon Lex still can't understand, it sends the message specified in abortStatement .
            messages (list) -- [REQUIRED]An array of objects, each of which provides a message string and its type. You can specify the message string in plain text or in Speech Synthesis Markup Language (SSML).
            (dict) --The message object that provides the message text and its type.
            contentType (string) -- [REQUIRED]The content type of the message string.
            content (string) -- [REQUIRED]The text of the message.
            
            maxAttempts (integer) -- [REQUIRED]The number of times to prompt the user for information.
            responseCard (string) --A response card. Amazon Lex uses this prompt at runtime, in the PostText API response. It substitutes session attributes and slot values for placeholders in the response card. For more information, see ex-resp-card .
            

    :type abortStatement: dict
    :param abortStatement: When Amazon Lex can't understand the user's input in context, it tries to elicit the information a few times. After that, Amazon Lex sends the message defined in abortStatement to the user, and then aborts the conversation. To set the number of retries, use the valueElicitationPrompt field for the slot type.
            For example, in a pizza ordering bot, Amazon Lex might ask a user 'What type of crust would you like?' If the user's response is not one of the expected responses (for example, 'thin crust, 'deep dish,' etc.), Amazon Lex tries to elicit a correct response a few more times.
            For example, in a pizza ordering application, OrderPizza might be one of the intents. This intent might require the CrustType slot. You specify the valueElicitationPrompt field when you create the CrustType slot.
            messages (list) -- [REQUIRED]A collection of message objects.
            (dict) --The message object that provides the message text and its type.
            contentType (string) -- [REQUIRED]The content type of the message string.
            content (string) -- [REQUIRED]The text of the message.
            
            responseCard (string) --At runtime, if the client is using the API, Amazon Lex includes the response card in the response. It substitutes all of the session attributes and slot values for placeholders in the response card.
            

    :type idleSessionTTLInSeconds: integer
    :param idleSessionTTLInSeconds: The maximum time in seconds that Amazon Lex retains the data gathered in a conversation.
            A user interaction session remains active for the amount of time specified. If no conversation occurs during this time, the session expires and Amazon Lex deletes any data provided before the timeout.
            For example, suppose that a user chooses the OrderPizza intent, but gets sidetracked halfway through placing an order. If the user doesn't complete the order within the specified time, Amazon Lex discards the slot information that it gathered, and the user must start over.
            If you don't include the idleSessionTTLInSeconds element in a PutBot operation request, Amazon Lex uses the default value. This is also true if the request replaces an existing bot.
            The default is 300 seconds (5 minutes).
            

    :type voiceId: string
    :param voiceId: The Amazon Polly voice ID that you want Amazon Lex to use for voice interactions with the user. The locale configured for the voice must match the locale of the bot. For more information, see Voice in the Amazon Polly Developer Guide .

    :type checksum: string
    :param checksum: Identifies a specific revision of the $LATEST version.
            When you create a new bot, leave the checksum field blank. If you specify a checksum you get a BadRequestException exception.
            When you want to update a bot, set the checksum field to the checksum of the most recent revision of the $LATEST version. If you don't specify the checksum field, or if the checksum does not match the $LATEST version, you get a PreconditionFailedException exception.
            

    :type processBehavior: string
    :param processBehavior: If you set the processBehavior element to Build , Amazon Lex builds the bot so that it can be run. If you set the element to Save Amazon Lex saves the bot, but doesn't build it.
            If you don't specify this value, the default value is Save .
            

    :type locale: string
    :param locale: [REQUIRED]
            Specifies the target locale for the bot. Any intent used in the bot must be compatible with the locale of the bot.
            The default is en-US .
            

    :type childDirected: boolean
    :param childDirected: [REQUIRED]
            For each Amazon Lex bot created with the Amazon Lex Model Building Service, you must specify whether your use of Amazon Lex is related to a website, program, or other application that is directed or targeted, in whole or in part, to children under age 13 and subject to the Children's Online Privacy Protection Act (COPPA) by specifying true or false in the childDirected field. By specifying true in the childDirected field, you confirm that your use of Amazon Lex is related to a website, program, or other application that is directed or targeted, in whole or in part, to children under age 13 and subject to COPPA. By specifying false in the childDirected field, you confirm that your use of Amazon Lex is not related to a website, program, or other application that is directed or targeted, in whole or in part, to children under age 13 and subject to COPPA. You may not specify a default value for the childDirected field that does not accurately reflect whether your use of Amazon Lex is related to a website, program, or other application that is directed or targeted, in whole or in part, to children under age 13 and subject to COPPA.
            If your use of Amazon Lex relates to a website, program, or other application that is directed in whole or in part, to children under age 13, you must obtain any required verifiable parental consent under COPPA. For information regarding the use of Amazon Lex in connection with websites, programs, or other applications that are directed or targeted, in whole or in part, to children under age 13, see the Amazon Lex FAQ.
            

    :rtype: dict
    :return: {
        'name': 'string',
        'description': 'string',
        'intents': [
            {
                'intentName': 'string',
                'intentVersion': 'string'
            },
        ],
        'clarificationPrompt': {
            'messages': [
                {
                    'contentType': 'PlainText'|'SSML',
                    'content': 'string'
                },
            ],
            'maxAttempts': 123,
            'responseCard': 'string'
        },
        'abortStatement': {
            'messages': [
                {
                    'contentType': 'PlainText'|'SSML',
                    'content': 'string'
                },
            ],
            'responseCard': 'string'
        },
        'status': 'BUILDING'|'READY'|'FAILED'|'NOT_BUILT',
        'failureReason': 'string',
        'lastUpdatedDate': datetime(2015, 1, 1),
        'createdDate': datetime(2015, 1, 1),
        'idleSessionTTLInSeconds': 123,
        'voiceId': 'string',
        'checksum': 'string',
        'version': 'string',
        'locale': 'en-US',
        'childDirected': True|False
    }
    
    
    """
    pass