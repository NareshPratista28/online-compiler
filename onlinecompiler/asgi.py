"""
ASGI config for onlinecompiler project.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'onlinecompiler.settings')

application = get_asgi_application()
