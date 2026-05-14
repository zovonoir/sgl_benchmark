"""Allow running as: python -m python_version"""

import sys
from .cli import main

sys.exit(main())
