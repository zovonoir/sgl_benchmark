"""Allow running as: python3 -m vllm_bench."""

import sys

from .cli import main


sys.exit(main())

