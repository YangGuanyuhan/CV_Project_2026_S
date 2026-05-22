# Contributing Guidelines

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YangGuanyuhan/CV_Project_2026_S.git
   cd CV_Project_2026_S
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

## Branch Naming Convention

Use the following branch naming convention:

- `feat/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Adding or updating tests

Examples:
- `feat/swin-backbone`
- `fix/coco-dataloader`
- `docs/update-readme`

## Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Examples

```
feat(models): add Swin Transformer backbone
fix(datasets): correct bounding box format in COCO loader
docs(readme): update installation instructions
refactor(engine): simplify training loop
test(utils): add unit tests for box_ops
```

## Pull Request Process

1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feat/your-feature
   ```

2. Make your changes and commit them following the commit convention.

3. Push your branch and create a Pull Request:
   ```bash
   git push origin feat/your-feature
   ```

4. Fill in the PR template with:
   - Description of changes
   - Related issue (if applicable)
   - Type of change
   - Checklist completion

5. Request review from at least one team member.

6. Address any review comments.

7. Once approved, merge into `develop`.

## Code Style

We use the following tools to maintain code quality:

- **Ruff**: For linting and formatting
- **Type Hints**: Use Python type hints for function signatures

### Running Code Style Checks

```bash
# Check for linting issues
ruff check .

# Format code
ruff format .
```

## Testing

Write tests for new features and bug fixes:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_box_ops.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions and classes
- Update relevant documentation in `docs/` directory

## Questions?

If you have any questions, feel free to open an issue or contact the team members.
