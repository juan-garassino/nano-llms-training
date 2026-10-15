import os
import sys

# Make `import nano_moe` work regardless of the directory pytest is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
