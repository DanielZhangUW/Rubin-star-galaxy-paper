# Overleaf GitHub Setup Notes

This repository is designed for the Overleaf workflow:

1. Push this manuscript-only repository to GitHub.
2. In Overleaf, create a new project using `New Project -> Import from GitHub`.
3. Select this manuscript repository.
4. Invite collaborators from Overleaf.
5. For ongoing work:
   - edit locally,
   - run `git add`, `git commit`, and `git push` to GitHub,
   - pull/sync in Overleaf when needed.

The main analysis repository should not be connected directly to Overleaf because it contains code, outputs, and private-data-related structure. This manuscript repository is intentionally small and clean.

## Manual GitHub Creation Commands

If GitHub CLI is unavailable, create an empty private GitHub repository named `Rubin-star-galaxy-paper`, then run:

```bash
cd /path/to/Rubin-star-galaxy-paper
git remote add origin https://github.com/<your-user-or-org>/Rubin-star-galaxy-paper.git
git push -u origin main
```
