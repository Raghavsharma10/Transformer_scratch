def checkUpgradePaths(self):
        """
        Check that all of the accumulated old Item types have a way to get
        from their current version to the latest version.

        @raise axiom.errors.NoUpgradePathAvailable: for any, and all, Items
            that do not have a valid upgrade path
        """
        cantUpgradeErrors = []

        for oldVersion in self._oldTypesRemaining:
            # We have to be able to get from oldVersion.schemaVersion to
            # the most recent type.

            currentType = _typeNameToMostRecentClass.get(
                oldVersion.typeName, None)

            if currentType is None:
                # There isn't a current version of this type; it's entirely
                # legacy, will be upgraded by deleting and replacing with
                # something else.
                continue

            typeInQuestion = oldVersion.typeName
            upgver = oldVersion.schemaVersion

            while upgver < currentType.schemaVersion:
                # Do we have enough of the schema present to upgrade?
                if ((typeInQuestion, upgver)
                    not in _upgradeRegistry):
                    cantUpgradeErrors.append(
                        "No upgrader present for %s (%s) from %d to %d" % (
                            typeInQuestion, qual(currentType), upgver,
                            upgver + 1))

                # Is there a type available for each upgrader version?
                if upgver+1 != currentType.schemaVersion:
                    if (typeInQuestion, upgver+1) not in _legacyTypes:
                        cantUpgradeErrors.append(
                            "Type schema required for upgrade missing:"
                            " %s version %d" % (
                                typeInQuestion, upgver+1))
                upgver += 1

            if cantUpgradeErrors:
                raise NoUpgradePathAvailable('\n    '.join(cantUpgradeErrors))