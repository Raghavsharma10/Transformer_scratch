def update_user_pool(UserPoolId=None, Policies=None, LambdaConfig=None, AutoVerifiedAttributes=None, SmsVerificationMessage=None, EmailVerificationMessage=None, EmailVerificationSubject=None, SmsAuthenticationMessage=None, MfaConfiguration=None, DeviceConfiguration=None, EmailConfiguration=None, SmsConfiguration=None, UserPoolTags=None, AdminCreateUserConfig=None):
    """
    Updates the specified user pool with the specified attributes.
    See also: AWS API Documentation
    
    
    :example: response = client.update_user_pool(
        UserPoolId='string',
        Policies={
            'PasswordPolicy': {
                'MinimumLength': 123,
                'RequireUppercase': True|False,
                'RequireLowercase': True|False,
                'RequireNumbers': True|False,
                'RequireSymbols': True|False
            }
        },
        LambdaConfig={
            'PreSignUp': 'string',
            'CustomMessage': 'string',
            'PostConfirmation': 'string',
            'PreAuthentication': 'string',
            'PostAuthentication': 'string',
            'DefineAuthChallenge': 'string',
            'CreateAuthChallenge': 'string',
            'VerifyAuthChallengeResponse': 'string'
        },
        AutoVerifiedAttributes=[
            'phone_number'|'email',
        ],
        SmsVerificationMessage='string',
        EmailVerificationMessage='string',
        EmailVerificationSubject='string',
        SmsAuthenticationMessage='string',
        MfaConfiguration='OFF'|'ON'|'OPTIONAL',
        DeviceConfiguration={
            'ChallengeRequiredOnNewDevice': True|False,
            'DeviceOnlyRememberedOnUserPrompt': True|False
        },
        EmailConfiguration={
            'SourceArn': 'string',
            'ReplyToEmailAddress': 'string'
        },
        SmsConfiguration={
            'SnsCallerArn': 'string',
            'ExternalId': 'string'
        },
        UserPoolTags={
            'string': 'string'
        },
        AdminCreateUserConfig={
            'AllowAdminCreateUserOnly': True|False,
            'UnusedAccountValidityDays': 123,
            'InviteMessageTemplate': {
                'SMSMessage': 'string',
                'EmailMessage': 'string',
                'EmailSubject': 'string'
            }
        }
    )
    
    
    :type UserPoolId: string
    :param UserPoolId: [REQUIRED]
            The user pool ID for the user pool you want to update.
            

    :type Policies: dict
    :param Policies: A container with the policies you wish to update in a user pool.
            PasswordPolicy (dict) --A container for information about the user pool password policy.
            MinimumLength (integer) --The minimum length of the password policy that you have set. Cannot be less than 6.
            RequireUppercase (boolean) --In the password policy that you have set, refers to whether you have required users to use at least one uppercase letter in their password.
            RequireLowercase (boolean) --In the password policy that you have set, refers to whether you have required users to use at least one lowercase letter in their password.
            RequireNumbers (boolean) --In the password policy that you have set, refers to whether you have required users to use at least one number in their password.
            RequireSymbols (boolean) --In the password policy that you have set, refers to whether you have required users to use at least one symbol in their password.
            
            

    :type LambdaConfig: dict
    :param LambdaConfig: The AWS Lambda configuration information from the request to update the user pool.
            PreSignUp (string) --A pre-registration AWS Lambda trigger.
            CustomMessage (string) --A custom Message AWS Lambda trigger.
            PostConfirmation (string) --A post-confirmation AWS Lambda trigger.
            PreAuthentication (string) --A pre-authentication AWS Lambda trigger.
            PostAuthentication (string) --A post-authentication AWS Lambda trigger.
            DefineAuthChallenge (string) --Defines the authentication challenge.
            CreateAuthChallenge (string) --Creates an authentication challenge.
            VerifyAuthChallengeResponse (string) --Verifies the authentication challenge response.
            

    :type AutoVerifiedAttributes: list
    :param AutoVerifiedAttributes: The attributes that are automatically verified when the Amazon Cognito service makes a request to update user pools.
            (string) --
            

    :type SmsVerificationMessage: string
    :param SmsVerificationMessage: A container with information about the SMS verification message.

    :type EmailVerificationMessage: string
    :param EmailVerificationMessage: The contents of the email verification message.

    :type EmailVerificationSubject: string
    :param EmailVerificationSubject: The subject of the email verification message.

    :type SmsAuthenticationMessage: string
    :param SmsAuthenticationMessage: The contents of the SMS authentication message.

    :type MfaConfiguration: string
    :param MfaConfiguration: Can be one of the following values:
            OFF - MFA tokens are not required and cannot be specified during user registration.
            ON - MFA tokens are required for all user registrations. You can only specify required when you are initially creating a user pool.
            OPTIONAL - Users have the option when registering to create an MFA token.
            

    :type DeviceConfiguration: dict
    :param DeviceConfiguration: Device configuration.
            ChallengeRequiredOnNewDevice (boolean) --Indicates whether a challenge is required on a new device. Only applicable to a new device.
            DeviceOnlyRememberedOnUserPrompt (boolean) --If true, a device is only remembered on user prompt.
            

    :type EmailConfiguration: dict
    :param EmailConfiguration: Email configuration.
            SourceArn (string) --The Amazon Resource Name (ARN) of the email source.
            ReplyToEmailAddress (string) --The REPLY-TO email address.
            

    :type SmsConfiguration: dict
    :param SmsConfiguration: SMS configuration.
            SnsCallerArn (string) -- [REQUIRED]The Amazon Resource Name (ARN) of the Amazon Simple Notification Service (SNS) caller.
            ExternalId (string) --The external ID.
            

    :type UserPoolTags: dict
    :param UserPoolTags: The cost allocation tags for the user pool. For more information, see Adding Cost Allocation Tags to Your User Pool
            (string) --
            (string) --
            

    :type AdminCreateUserConfig: dict
    :param AdminCreateUserConfig: The configuration for AdminCreateUser requests.
            AllowAdminCreateUserOnly (boolean) --Set to True if only the administrator is allowed to create user profiles. Set to False if users can sign themselves up via an app.
            UnusedAccountValidityDays (integer) --The user account expiration limit, in days, after which the account is no longer usable. To reset the account after that time limit, you must call AdminCreateUser again, specifying 'RESEND' for the MessageAction parameter. The default value for this parameter is 7.
            InviteMessageTemplate (dict) --The message template to be used for the welcome message to new users.
            SMSMessage (string) --The message template for SMS messages.
            EmailMessage (string) --The message template for email messages.
            EmailSubject (string) --The subject line for email messages.
            
            

    :rtype: dict
    :return: {}
    
    
    """
    pass