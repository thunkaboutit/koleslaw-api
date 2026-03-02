# Contributing to ThunkAboutIt Demos

Thank you for your interest in contributing! This guide will help you understand our development workflow and commit
conventions.

## Table of Contents

- [Getting Started](#getting-started)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Development Workflow](#development-workflow)

---

## Getting Started

1. **Fork the repository** and clone it locally
2. **Create a feature branch** from `main` (or `develop` if applicable)
3. **Make your changes** following our code style guidelines
4. **Write tests** for new features or bug fixes
5. **Commit your changes** using conventional commit format
6. **Push to your fork** and submit a pull request

---

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification for clear and structured commit
history. This helps with:

- **Automated versioning** and changelog generation
- **Clear communication** of changes to the team
- **Easy navigation** through project history
- **Semantic versioning** automation

### Commit Message Structure

```
<type>[optional scope][optional !]: <description>

[optional body]

[optional footer(s)]
```

### Quick Examples

```
feat: add user authentication
fix(api): resolve null pointer exception in user service
feat!: redesign authentication API
feat(auth)!: rework API to use OAuth2
chore(deps): update Spring Boot to 3.2.0
docs: update API documentation for v2 endpoints
```

### Commit Types

| Type       | Description              | When to Use                           | Semver Impact |
|------------|--------------------------|---------------------------------------|---------------|
| `feat`     | A new feature            | Adding new functionality              | MINOR         |
| `fix`      | A bug fix                | Fixing a defect or issue              | PATCH         |
| `docs`     | Documentation changes    | README, comments, guides              | -             |
| `style`    | Code style changes       | Formatting, whitespace, semicolons    | -             |
| `refactor` | Code refactoring         | Restructuring without behavior change | -             |
| `perf`     | Performance improvements | Optimizations                         | PATCH         |
| `test`     | Test changes             | Adding or updating tests              | -             |
| `build`    | Build system changes     | Webpack, Maven, Gradle, npm           | -             |
| `ci`       | CI/CD changes            | GitHub Actions, Jenkins, Travis       | -             |
| `chore`    | Maintenance tasks        | Tooling, configs, dependencies        | -             |
| `revert`   | Revert a previous commit | Undoing changes                       | -             |

**Most commonly used**: `feat`, `fix`, `chore`, `docs`

### Type Details

#### `feat` - New Features

Use when adding new functionality to the codebase.

**Examples:**

```
feat: add pagination to user list endpoint
feat(api): implement GraphQL query support
feat(ui): add dark mode toggle
feat(auth): integrate two-factor authentication
```

#### `fix` - Bug Fixes

Use when fixing defects, errors, or unintended behavior.

**Examples:**

```
fix: prevent race condition in payment processing
fix(database): resolve connection pool exhaustion
fix(ui): correct alignment issue on mobile devices
fix(api): handle null values in user profile endpoint
```

#### `docs` - Documentation

Use for changes to documentation files, code comments, or README updates.

**Examples:**

```
docs: add API usage examples to README
docs(contributing): update commit message guidelines
docs: fix typos in installation guide
docs(api): add OpenAPI specification
```

#### `style` - Code Style

Use for formatting changes that don't affect code behavior (whitespace, indentation, etc.).

**Examples:**

```
style: format code with prettier
style(java): apply Google Java Style Guide
style: remove trailing whitespace
style(css): organize imports alphabetically
```

#### `refactor` - Code Refactoring

Use when restructuring code without changing its external behavior.

**Examples:**

```
refactor: extract user validation into separate service
refactor(auth): simplify token generation logic
refactor: replace nested if statements with guard clauses
refactor(database): consolidate duplicate queries
```

#### `perf` - Performance Improvements

Use when optimizing code for better performance.

**Examples:**

```
perf: implement caching for frequently accessed data
perf(database): add index on user email column
perf(api): reduce payload size by 40%
perf: lazy load images on user profile page
```

#### `test` - Tests

Use when adding, updating, or fixing tests.

**Examples:**

```
test: add integration tests for payment service
test(auth): increase coverage for login edge cases
test: fix flaky test in order processing
test(e2e): add tests for checkout flow
```

#### `build` - Build System

Use for changes to build configurations, dependencies, or build scripts.

**Examples:**

```
build: upgrade to Node 20 LTS
build(maven): update compiler plugin to 3.11.0
build: add webpack bundle analyzer
build(gradle): configure multi-module project structure
```

#### `ci` - Continuous Integration

Use for CI/CD pipeline, workflow, or automation changes.

**Examples:**

```
ci: add automated security scanning
ci(github): configure dependabot for dependency updates
ci: reduce test execution time by parallelization
ci(jenkins): add staging deployment pipeline
```

#### `chore` - Maintenance

Use for routine tasks, tooling, or configurations that don't modify source code or tests.

**Examples:**

```
chore: update .gitignore to exclude log files
chore(deps): bump lodash from 4.17.19 to 4.17.21
chore: configure ESLint for TypeScript
chore(release): prepare for v2.0.0
```

#### `revert` - Revert

Use when reverting a previous commit.

**Format:**

```
revert: <header of reverted commit>

This reverts commit <hash>.
```

**Example:**

```
revert: feat: add user authentication

This reverts commit a1b2c3d4e5f6.
```

### Scope (Optional)

The scope provides additional context about which part of the codebase was changed.

**Format:** `type(scope): description`

**Common scopes:**

- `api` - API changes
- `ui` / `frontend` - User interface
- `backend` - Backend logic
- `database` / `db` - Database related
- `auth` - Authentication/authorization
- `deps` - Dependencies
- `config` - Configuration files
- `docs` - Documentation
- Component/module names: `user-service`, `payment`, `dashboard`

**Examples:**

```
feat(api): add rate limiting middleware
fix(auth): resolve token expiration issue
chore(deps): update all non-breaking dependencies
refactor(payment): simplify checkout logic
test(user-service): add edge case coverage
```

### Breaking Changes

Breaking changes **MUST** be indicated by:

1. Adding `!` after the type/scope
2. Including `BREAKING CHANGE:` in the commit footer

**Format with `!`:**

```
feat!: remove deprecated authentication endpoints
fix(api)!: change response format to match OpenAPI spec
```

**Format with footer:**

```
feat(api): redesign user authentication

BREAKING CHANGE: The /auth/login endpoint now requires email instead of username.
Migration guide: https://docs.example.com/migration-v2
```

**Full example:**

```
feat(api)!: implement v2 REST API

Redesigned all endpoints to follow RESTful conventions.
Added HATEOAS links to all responses.

BREAKING CHANGE: All v1 endpoints are deprecated and will be removed in v3.0.0.
Clients must update to use /api/v2/* endpoints.
See migration guide: https://docs.example.com/v1-to-v2
```

### Commit Message Body

The body is **optional** and provides additional context. Use it for:

- **Motivation** for the change
- **Comparison** with previous behavior
- **Impact** explanation
- **Implementation details**

**Requirements:**

- Separate from subject with a **blank line**
- Wrap at **72 characters**
- Use **imperative mood** ("add feature" not "added feature")

**Example:**

```
fix(payment): prevent duplicate charge on retry

The previous implementation didn't check for existing charges
before processing retries, leading to duplicate billing when
users experienced network timeouts.

This fix adds idempotency key validation and charge lookup
before processing any payment retry attempts.

Resolves #123
```

### Commit Message Footer

The footer is **optional** and used for:

- **Issue references**: `Closes #123`, `Fixes #456`, `Resolves #789`
- **Breaking changes**: `BREAKING CHANGE: description`
- **Co-authors**: `Co-authored-by: Name <email>`
- **Reviewed by**: `Reviewed-by: Name <email>`

**Examples:**

```
feat: add export functionality

Users can now export their data in CSV or JSON format.

Closes #234
Co-authored-by: Jane Doe <jane@example.com>
```

```
fix(security): patch XSS vulnerability in comment system

BREAKING CHANGE: HTML is now escaped in all user comments.
Custom formatting tags are no longer supported.

Fixes #567
```

### Complete Commit Examples

#### Simple commit

```
feat: add user profile page
```

#### Commit with scope

```
fix(auth): resolve session timeout issue
```

#### Commit with body

```
refactor(database): optimize query performance

Replaced N+1 queries with batch loading using DataLoader.
Reduced average response time from 800ms to 120ms.

Closes #456
```

#### Breaking change commit

```
feat(api)!: redesign authentication flow

Migrated from session-based to JWT token authentication.
All clients must update their authentication logic.

BREAKING CHANGE: Session cookies are no longer supported.
Use Bearer tokens in Authorization header instead.

Migration guide: https://docs.example.com/auth-migration

Closes #789
Co-authored-by: John Smith <john@example.com>
```

#### Multi-issue commit

```
fix(ui): resolve layout issues on mobile

- Fixed navbar overflow on small screens
- Corrected z-index layering in modal dialogs
- Improved touch target sizes for buttons

Fixes #123, #124, #125
```

### Commit Message Best Practices

✅ **DO:**

- Use **imperative mood**: "add feature" not "added feature" or "adds feature"
- Keep subject line **under 72 characters**
- Start subject with **lowercase** (unless proper noun)
- **Don't end** subject with a period
- Use body to **explain what and why**, not how
- Reference **issues and PRs** in footer
- Write **clear, specific** descriptions

❌ **DON'T:**

- Use generic messages like "fix bug" or "update code"
- Include file names in subject (use scope instead)
- Mix multiple unrelated changes in one commit
- Use past tense ("fixed" or "added")
- Write subjects longer than 72 characters

**Bad examples:**

```
❌ fixed the bug
❌ Update user.java and add new feature
❌ WIP
❌ feat: Added new feature for the user authentication system that allows users to login
```

**Good examples:**

```
✅ fix(auth): resolve null pointer in login validation
✅ feat(user): add email verification
✅ docs: update installation instructions
✅ refactor: extract payment logic into separate service
```

### Validation Tools

We recommend using commit message linters to validate your commits:

**For Node.js projects:**

```bash
npm install --save-dev @commitlint/cli @commitlint/config-conventional
```

**For Git hooks:**

```bash
npm install --save-dev husky
npx husky init
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit ${1}'
```

**Manual validation:**

```bash
# Validate last commit
git log -1 --pretty=%B | npx commitlint
```

---

## Pull Request Process

### Before Submitting

1. ✅ **Update documentation** if you changed APIs or added features
2. ✅ **Add/update tests** to cover your changes
3. ✅ **Run the test suite** and ensure all tests pass
4. ✅ **Follow code style** guidelines
5. ✅ **Rebase on latest main** to avoid merge conflicts
6. ✅ **Use conventional commits** for all commit messages

### PR Title

Use conventional commit format for PR titles:

```
feat(api): add user search endpoint
fix(auth): resolve token refresh race condition
docs: update contributing guidelines
```

### PR Description Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

Describe the tests you ran and how to reproduce them.

## Checklist

- [ ] My code follows the code style of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Related Issues

Closes #123
Fixes #456
```

### Review Process

1. At least **one approval** required before merging
2. All **CI checks must pass**
3. No **unresolved conversations**
4. **Rebase and squash** commits if needed for clean history
5. Use **squash and merge** for feature branches (single commit to main)
6. Use **rebase and merge** for keeping linear history (if preferred)

---

## Code Style Guidelines

### General Principles

- **Readability** over cleverness
- **Consistency** with existing codebase
- **Self-documenting** code with clear naming
- **DRY (Don't Repeat Yourself)**
- **SOLID principles**

### Language-Specific Guides

- **Java**: Follow [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)
- **JavaScript/TypeScript**: Follow [Airbnb Style Guide](https://github.com/airbnb/javascript)
- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Go**: Follow [Effective Go](https://golang.org/doc/effective_go.html)

### Code Review Focus Areas

Reviewers should focus on:

- **Logic errors** and edge cases
- **Performance** implications
- **Security** vulnerabilities
- **Test coverage** and quality
- **Documentation** clarity
- **API design** and usability

---

## Development Workflow

### Branch Naming

Use descriptive branch names with type prefix:

```
feature/user-authentication
fix/payment-processing-bug
docs/api-documentation-update
refactor/database-query-optimization
```

### Workflow Steps

1. **Create feature branch**: `git checkout -b feature/my-feature`
2. **Make changes** in small, logical commits
3. **Keep commits focused**: One concern per commit
4. **Write tests** as you develop
5. **Commit with conventional format**
6. **Push to your fork**: `git push origin feature/my-feature`
7. **Open pull request** with clear description
8. **Address review feedback** in new commits
9. **Squash commits** if requested before merge

### Commit Frequency

- **Commit often** during development (helps with code review)
- **Keep commits atomic**: Each commit should be a complete, working change
- **Don't commit broken code** to shared branches
- **Can squash** before merging if you have many small commits

---

## Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- [Git Best Practices](https://www.git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project)

---

## Questions or Issues?

If you have questions about contributing:

- Open a **discussion** in the repository
- Reach out in our **community chat** [link]
- Contact maintainers at **[email]**

Thank you for contributing to ThunkAboutIt Demos! 🎉