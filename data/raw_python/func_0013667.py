def send_email(Source=None, Destination=None, Message=None, ReplyToAddresses=None, ReturnPath=None, SourceArn=None, ReturnPathArn=None, Tags=None, ConfigurationSetName=None):
    """
    Composes an email message based on input data, and then immediately queues the message for sending.
    There are several important points to know about SendEmail :
    See also: AWS API Documentation
    
    Examples
    The following example sends a formatted email:
    Expected Output:
    
    :example: response = client.send_email(
        Source='string',
        Destination={
            'ToAddresses': [
                'string',
            ],
            'CcAddresses': [
                'string',
            ],
            'BccAddresses': [
                'string',
            ]
        },
        Message={
            'Subject': {
                'Data': 'string',
                'Charset': 'string'
            },
            'Body': {
                'Text': {
                    'Data': 'string',
                    'Charset': 'string'
                },
                'Html': {
                    'Data': 'string',
                    'Charset': 'string'
                }
            }
        },
        ReplyToAddresses=[
            'string',
        ],
        ReturnPath='string',
        SourceArn='string',
        ReturnPathArn='string',
        Tags=[
            {
                'Name': 'string',
                'Value': 'string'
            },
        ],
        ConfigurationSetName='string'
    )
    
    
    :type Source: string
    :param Source: [REQUIRED]
            The email address that is sending the email. This email address must be either individually verified with Amazon SES, or from a domain that has been verified with Amazon SES. For information about verifying identities, see the Amazon SES Developer Guide .
            If you are sending on behalf of another user and have been permitted to do so by a sending authorization policy, then you must also specify the SourceArn parameter. For more information about sending authorization, see the Amazon SES Developer Guide .
            In all cases, the email address must be 7-bit ASCII. If the text must contain any other characters, then you must use MIME encoded-word syntax (RFC 2047) instead of a literal string. MIME encoded-word syntax uses the following form: =?charset?encoding?encoded-text?= . For more information, see RFC 2047 .
            

    :type Destination: dict
    :param Destination: [REQUIRED]
            The destination for this email, composed of To:, CC:, and BCC: fields.
            ToAddresses (list) --The To: field(s) of the message.
            (string) --
            CcAddresses (list) --The CC: field(s) of the message.
            (string) --
            BccAddresses (list) --The BCC: field(s) of the message.
            (string) --
            

    :type Message: dict
    :param Message: [REQUIRED]
            The message to be sent.
            Subject (dict) -- [REQUIRED]The subject of the message: A short summary of the content, which will appear in the recipient's inbox.
            Data (string) -- [REQUIRED]The textual data of the content.
            Charset (string) --The character set of the content.
            Body (dict) -- [REQUIRED]The message body.
            Text (dict) --The content of the message, in text format. Use this for text-based email clients, or clients on high-latency networks (such as mobile devices).
            Data (string) -- [REQUIRED]The textual data of the content.
            Charset (string) --The character set of the content.
            Html (dict) --The content of the message, in HTML format. Use this for email clients that can process HTML. You can include clickable links, formatted text, and much more in an HTML message.
            Data (string) -- [REQUIRED]The textual data of the content.
            Charset (string) --The character set of the content.
            
            

    :type ReplyToAddresses: list
    :param ReplyToAddresses: The reply-to email address(es) for the message. If the recipient replies to the message, each reply-to address will receive the reply.
            (string) --
            

    :type ReturnPath: string
    :param ReturnPath: The email address to which bounces and complaints are to be forwarded when feedback forwarding is enabled. If the message cannot be delivered to the recipient, then an error message will be returned from the recipient's ISP; this message will then be forwarded to the email address specified by the ReturnPath parameter. The ReturnPath parameter is never overwritten. This email address must be either individually verified with Amazon SES, or from a domain that has been verified with Amazon SES.

    :type SourceArn: string
    :param SourceArn: This parameter is used only for sending authorization. It is the ARN of the identity that is associated with the sending authorization policy that permits you to send for the email address specified in the Source parameter.
            For example, if the owner of example.com (which has ARN arn:aws:ses:us-east-1:123456789012:identity/example.com ) attaches a policy to it that authorizes you to send from user@example.com , then you would specify the SourceArn to be arn:aws:ses:us-east-1:123456789012:identity/example.com , and the Source to be user@example.com .
            For more information about sending authorization, see the Amazon SES Developer Guide .
            

    :type ReturnPathArn: string
    :param ReturnPathArn: This parameter is used only for sending authorization. It is the ARN of the identity that is associated with the sending authorization policy that permits you to use the email address specified in the ReturnPath parameter.
            For example, if the owner of example.com (which has ARN arn:aws:ses:us-east-1:123456789012:identity/example.com ) attaches a policy to it that authorizes you to use feedback@example.com , then you would specify the ReturnPathArn to be arn:aws:ses:us-east-1:123456789012:identity/example.com , and the ReturnPath to be feedback@example.com .
            For more information about sending authorization, see the Amazon SES Developer Guide .
            

    :type Tags: list
    :param Tags: A list of tags, in the form of name/value pairs, to apply to an email that you send using SendEmail . Tags correspond to characteristics of the email that you define, so that you can publish email sending events.
            (dict) --Contains the name and value of a tag that you can provide to SendEmail or SendRawEmail to apply to an email.
            Message tags, which you use with configuration sets, enable you to publish email sending events. For information about using configuration sets, see the Amazon SES Developer Guide .
            Name (string) -- [REQUIRED]The name of the tag. The name must:
            Contain only ASCII letters (a-z, A-Z), numbers (0-9), underscores (_), or dashes (-).
            Contain less than 256 characters.
            Value (string) -- [REQUIRED]The value of the tag. The value must:
            Contain only ASCII letters (a-z, A-Z), numbers (0-9), underscores (_), or dashes (-).
            Contain less than 256 characters.
            
            

    :type ConfigurationSetName: string
    :param ConfigurationSetName: The name of the configuration set to use when you send an email using SendEmail .

    :rtype: dict
    :return: {
        'MessageId': 'string'
    }
    
    
    :returns: 
    Source (string) -- [REQUIRED]
    The email address that is sending the email. This email address must be either individually verified with Amazon SES, or from a domain that has been verified with Amazon SES. For information about verifying identities, see the Amazon SES Developer Guide .
    If you are sending on behalf of another user and have been permitted to do so by a sending authorization policy, then you must also specify the SourceArn parameter. For more information about sending authorization, see the Amazon SES Developer Guide .
    In all cases, the email address must be 7-bit ASCII. If the text must contain any other characters, then you must use MIME encoded-word syntax (RFC 2047) instead of a literal string. MIME encoded-word syntax uses the following form: =?charset?encoding?encoded-text?= . For more information, see RFC 2047 .
    
    Destination (dict) -- [REQUIRED]
    The destination for this email, composed of To:, CC:, and BCC: fields.
    
    ToAddresses (list) --The To: field(s) of the message.
    
    (string) --
    
    
    CcAddresses (list) --The CC: field(s) of the message.
    
    (string) --
    
    
    BccAddresses (list) --The BCC: field(s) of the message.
    
    (string) --
    
    
    
    
    Message (dict) -- [REQUIRED]
    The message to be sent.
    
    Subject (dict) -- [REQUIRED]The subject of the message: A short summary of the content, which will appear in the recipient's inbox.
    
    Data (string) -- [REQUIRED]The textual data of the content.
    
    Charset (string) --The character set of the content.
    
    
    
    Body (dict) -- [REQUIRED]The message body.
    
    Text (dict) --The content of the message, in text format. Use this for text-based email clients, or clients on high-latency networks (such as mobile devices).
    
    Data (string) -- [REQUIRED]The textual data of the content.
    
    Charset (string) --The character set of the content.
    
    
    
    Html (dict) --The content of the message, in HTML format. Use this for email clients that can process HTML. You can include clickable links, formatted text, and much more in an HTML message.
    
    Data (string) -- [REQUIRED]The textual data of the content.
    
    Charset (string) --The character set of the content.
    
    
    
    
    
    
    
    ReplyToAddresses (list) -- The reply-to email address(es) for the message. If the recipient replies to the message, each reply-to address will receive the reply.
    
    (string) --
    
    
    ReturnPath (string) -- The email address to which bounces and complaints are to be forwarded when feedback forwarding is enabled. If the message cannot be delivered to the recipient, then an error message will be returned from the recipient's ISP; this message will then be forwarded to the email address specified by the ReturnPath parameter. The ReturnPath parameter is never overwritten. This email address must be either individually verified with Amazon SES, or from a domain that has been verified with Amazon SES.
    SourceArn (string) -- This parameter is used only for sending authorization. It is the ARN of the identity that is associated with the sending authorization policy that permits you to send for the email address specified in the Source parameter.
    For example, if the owner of example.com (which has ARN arn:aws:ses:us-east-1:123456789012:identity/example.com ) attaches a policy to it that authorizes you to send from user@example.com , then you would specify the SourceArn to be arn:aws:ses:us-east-1:123456789012:identity/example.com , and the Source to be user@example.com .
    For more information about sending authorization, see the Amazon SES Developer Guide .
    
    ReturnPathArn (string) -- This parameter is used only for sending authorization. It is the ARN of the identity that is associated with the sending authorization policy that permits you to use the email address specified in the ReturnPath parameter.
    For example, if the owner of example.com (which has ARN arn:aws:ses:us-east-1:123456789012:identity/example.com ) attaches a policy to it that authorizes you to use feedback@example.com , then you would specify the ReturnPathArn to be arn:aws:ses:us-east-1:123456789012:identity/example.com , and the ReturnPath to be feedback@example.com .
    For more information about sending authorization, see the Amazon SES Developer Guide .
    
    Tags (list) -- A list of tags, in the form of name/value pairs, to apply to an email that you send using SendEmail . Tags correspond to characteristics of the email that you define, so that you can publish email sending events.
    
    (dict) --Contains the name and value of a tag that you can provide to SendEmail or SendRawEmail to apply to an email.
    Message tags, which you use with configuration sets, enable you to publish email sending events. For information about using configuration sets, see the Amazon SES Developer Guide .
    
    Name (string) -- [REQUIRED]The name of the tag. The name must:
    
    Contain only ASCII letters (a-z, A-Z), numbers (0-9), underscores (_), or dashes (-).
    Contain less than 256 characters.
    
    
    Value (string) -- [REQUIRED]The value of the tag. The value must:
    
    Contain only ASCII letters (a-z, A-Z), numbers (0-9), underscores (_), or dashes (-).
    Contain less than 256 characters.
    
    
    
    
    
    
    ConfigurationSetName (string) -- The name of the configuration set to use when you send an email using SendEmail .
    
    """
    pass