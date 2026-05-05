import os

from .datasets import *

# Avoid importing heavy ops during data preparation.
if os.environ.get("MOMAD_LIGHT_IMPORT") != "1":
    from .models import *
    from .apis import *
    from .core.evaluation import *
