# Lint the Python code using ruff
[group('Dev')]
lint:
    uvx ruff@latest check .

# Format the Python code using ruff
[group('Dev')]
format:
    uvx ruff@latest format .
    uvx ruff@latest check . --fix
