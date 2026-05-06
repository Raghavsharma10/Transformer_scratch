def _forwards(apps, schema_editor):
    """
    Make sure that the MarkupItem model actually points
    to the correct proxy model, that implements the given language.
    """
    # Need to work on the actual models here.
    from fluent_contents.plugins.markup.models import LANGUAGE_MODEL_CLASSES
    from fluent_contents.plugins.markup.models import MarkupItem
    from django.contrib.contenttypes.models import ContentType

    ctype = ContentType.objects.get_for_model(MarkupItem)

    for language, proxy_model in LANGUAGE_MODEL_CLASSES.items():
        proxy_ctype = ContentType.objects.get_for_model(proxy_model, for_concrete_model=False)
        MarkupItem.objects.filter(
            polymorphic_ctype=ctype, language=language
        ).update(
            polymorphic_ctype=proxy_ctype
        )