with open("core/tier0.py", "r", encoding="utf-8") as f:
    content = f.read()

import re

# Fix backspaces by replacing with \b
content = content.replace('\x08', r'\b')

with open("core/tier0.py", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: Fixed \\b in core/tier0.py")
