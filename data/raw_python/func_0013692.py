def update_account_password_policy(MinimumPasswordLength=None, RequireSymbols=None, RequireNumbers=None, RequireUppercaseCharacters=None, RequireLowercaseCharacters=None, AllowUsersToChangePassword=None, MaxPasswordAge=None, PasswordReusePrevention=None, HardExpiry=None):
    """
    Updates the password policy settings for the AWS account.
    For more information about using a password policy, see Managing an IAM Password Policy in the IAM User Guide .
    See also: AWS API Documentation
    
    Examples
    The following command sets the password policy to require a minimum length of eight characters and to require one or more numbers in the password:
    Expected Output:
    
    :example: response = client.update_account_password_policy(
        MinimumPasswordLength=123,
        RequireSymbols=True|False,
        RequireNumbers=True|False,
        RequireUppercaseCharacters=True|False,
        RequireLowercaseCharacters=True|False,
        AllowUsersToChangePassword=True|False,
        MaxPasswordAge=123,
        PasswordReusePrevention=123,
        HardExpiry=True|False
    )
    
    
    :type MinimumPasswordLength: integer
    :param MinimumPasswordLength: The minimum number of characters allowed in an IAM user password.
            Default value: 6
            

    :type RequireSymbols: boolean
    :param RequireSymbols: Specifies whether IAM user passwords must contain at least one of the following non-alphanumeric characters:
            ! @ # $ % ^ amp; * ( ) _ + - = [ ] { } | '
            Default value: false
            

    :type RequireNumbers: boolean
    :param RequireNumbers: Specifies whether IAM user passwords must contain at least one numeric character (0 to 9).
            Default value: false
            

    :type RequireUppercaseCharacters: boolean
    :param RequireUppercaseCharacters: Specifies whether IAM user passwords must contain at least one uppercase character from the ISO basic Latin alphabet (A to Z).
            Default value: false
            

    :type RequireLowercaseCharacters: boolean
    :param RequireLowercaseCharacters: Specifies whether IAM user passwords must contain at least one lowercase character from the ISO basic Latin alphabet (a to z).
            Default value: false
            

    :type AllowUsersToChangePassword: boolean
    :param AllowUsersToChangePassword: Allows all IAM users in your account to use the AWS Management Console to change their own passwords. For more information, see Letting IAM Users Change Their Own Passwords in the IAM User Guide .
            Default value: false
            

    :type MaxPasswordAge: integer
    :param MaxPasswordAge: The number of days that an IAM user password is valid. The default value of 0 means IAM user passwords never expire.
            Default value: 0
            

    :type PasswordReusePrevention: integer
    :param PasswordReusePrevention: Specifies the number of previous passwords that IAM users are prevented from reusing. The default value of 0 means IAM users are not prevented from reusing previous passwords.
            Default value: 0
            

    :type HardExpiry: boolean
    :param HardExpiry: Prevents IAM users from setting a new password after their password has expired.
            Default value: false
            

    :return: response = client.update_account_password_policy(
        MinimumPasswordLength=8,
        RequireNumbers=True,
    )
    
    print(response)
    
    
    """
    pass