def do_mfa(self, args):
        """
        Enter a 6-digit MFA token. Nephele will execute the appropriate
        `aws` command line to authenticate that token. 

        mfa -h for more details
        """
        
        parser = CommandArgumentParser("mfa")
        parser.add_argument(dest='token',help='MFA token value');
        parser.add_argument("-p","--profile",dest='awsProfile',default=AwsConnectionFactory.instance.getProfile(),help='MFA token value');
        args = vars(parser.parse_args(args))

        token = args['token']
        awsProfile = args['awsProfile']
        arn = AwsConnectionFactory.instance.load_arn(awsProfile)

        credentials_command = ["aws","--profile",awsProfile,"--output","json","sts","get-session-token","--serial-number",arn,"--token-code",token]
        output = run_cmd(credentials_command) # Throws on non-zero exit :yey:

        credentials = json.loads("\n".join(output.stdout))['Credentials']
        AwsConnectionFactory.instance.setMfaCredentials(credentials,awsProfile)