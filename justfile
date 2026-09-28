# Lint the Python code using ruff
[group('Dev')]
lint:
    uvx ruff@latest check . --config .\ruff.toml

# Format the Python code using ruff
[group('Dev')]
format:
    uvx ruff@latest format . --config .\ruff.toml
    uvx ruff@latest check . --config .\ruff.toml --fix
