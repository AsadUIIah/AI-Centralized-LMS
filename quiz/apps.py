from django.apps import AppConfig


class QuizConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'quiz'

    def ready(self):
        # Load the fine-tuned QG model once when Django starts, not per-request.
        # Imported here (not at module top-level) to avoid loading heavy ML
        # libraries during management commands that don't need them (e.g.
        # makemigrations), and to avoid the "apps not ready yet" issues that
        # can happen with top-level imports in apps.py.
        from . import ml_utils
        ml_utils.load_models()