def deploy(self):
        """Upload code to AWS Lambda. To use this method, first, must set the zip file with code with
         `self.set_artefact(code=code)`. Check all lambdas in our config file or the functions passed in command line
         and exist in our config file. If the function is upload correctly, update/create versions, alias and
         triggers

        :return: True
        """
        lambdas_deployed = []
        for lambda_funcion in self.config.get_lambdas():
            start_deploy = not len(self.lambdas_to_deploy) or \
                           lambda_funcion["FunctionNameOrigin"] in self.lambdas_to_deploy

            if start_deploy:
                lambdas_deployed.append(lambda_funcion["FunctionName"])
                conf = lambda_funcion.get_deploy_conf()
                response = self.remote_get_lambda(**conf)
                if response:
                    remote_conf = response["Configuration"]

                    # TODO: Diferences sometimes not return all values, check it!
                    logger.info("Diferences:")
                    diffkeys = [k for k in remote_conf if
                                conf.get(k, False) != remote_conf.get(k, True) and k not in ['Code', ]]
                    for k in diffkeys:
                        logger.info((k, ':', conf.get(k, ""), '->', remote_conf.get(k, "")))

                    logger.info("START to update funcion {}".format(conf["FunctionName"]))
                    self.remote_update_conf_lambada(**conf)
                    result = self.remote_update_code_lambada(**conf)
                    logger.debug("Funcion {} updated {}".format(conf["FunctionName"], result))

                else:
                    logger.info("START to create funcion {}".format(lambda_funcion["FunctionName"]))
                    result = self.remote_create_lambada(**conf)
                    logger.debug("Funcion {} created {}".format(conf["FunctionName"], result))

                if self.is_client_result_ok(result):

                    # Check and publish version
                    version = "LATEST"
                    if self.config["deploy"].get("use_version", False):
                        logger.info("Publish new version of {} with conf {}".format(
                            lambda_funcion["FunctionName"],
                            json.dumps(conf, indent=4, sort_keys=True)
                        ))
                        result = self.remote_publish_version(**conf)
                        version = result["Version"]
                        logger.info("Published version {}: {}".format(
                            version,
                            json.dumps(result, indent=4, sort_keys=True)
                        ))

                    # Check and publish alias
                    if self.config["deploy"].get("use_alias", False):
                        alias_conf = {
                            "FunctionName": conf["FunctionName"],
                            "Description": conf["Description"],
                            "FunctionVersion": version,
                        }
                        if self.config.get_environment():
                            alias_conf.update({"Name": self.config.get_environment()})
                        else:
                            alias_conf.update({"Name": conf["FunctionName"]})

                        logger.info("Update alias of {} with conf {}".format(
                            lambda_funcion["FunctionName"],
                            json.dumps(alias_conf, indent=4, sort_keys=True)
                        ))
                        result = self.remote_update_alias(**alias_conf)
                        logger.info("Updated alias {}: {}".format(conf["FunctionName"],
                                                                  json.dumps(result, indent=4, sort_keys=True)
                                                                  ))

                    # Check and publish triggers
                    logger.info("Updating Triggers for fuction {}".format(lambda_funcion["FunctionName"]))
                    if lambda_funcion.get("triggers", False):
                        for trigger in lambda_funcion["triggers"].keys():
                            trigger_object = get_trigger(trigger, lambda_funcion, result["FunctionArn"])
                            trigger_object.put()

        if lambdas_deployed:
            logger.info("Deploy finished. Created/updated lambdas {}".format(", ".join(lambdas_deployed)))
        else:
            logger.info("No lambdas found to deploy")

        # TODO: check errors to return correct value
        return True