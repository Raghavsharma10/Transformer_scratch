# Learning Log

## Softmax implementation

### Python
- Softmax needs max-subtraction for numerical stability, to check its stability for higher values - thats why we 
added the second test with higher end values.
- Tests should check properties (sum=1), not exact values

### Git
- `git clone` already creates `.git` — never run `git init` after --> git clone be run when repo is created on github already -> git clone = mkdir + download + git init + remote setup

- Git tracks directories via nearest `.git`, not by intention, only in downwards direction of directories.
Therefore each project directory should have its git init, seperately. 

- If `git status` shows random files, you're in the wrong repo.

### Pytest
- ** Python packages need `__init__.py` -> If want a folder to behave as a module for import - __init__.py is necessary in that folder
- Always run pytest from repo root

### Mental models
- One file = one idea
- Commit after each trusted primitive
- Tests are contracts, not just checks