# Release Checklist

This checklist guides you through releasing a new version of greedytile to both crates.io and GitHub.

## Prerequisites

- [ ] All changes are committed locally
- [ ] You have cargo and gh CLI tools installed
- [ ] You're logged into crates.io (`cargo login`)

## Release Steps

### 1. Update Version in Cargo.toml

```bash
# Edit Cargo.toml and change version from "0.1.6" to "0.1.7"
# Or use sed:
sed -i 's/version = "0.1.6"/version = "0.1.7"/' Cargo.toml
```

### 2. Run Tests and Build

```bash
cargo test
cargo build --release
```

### 3. Commit All Changes

```bash
git add .
git commit -m "Release v0.1.7"
```

### 4. Push to GitHub (IMPORTANT: Do this BEFORE creating the tag!)

```bash
git push origin master
```

### 5. Create and Push Git Tag (ONLY after the commit is pushed)

```bash
git tag v0.1.7
git push origin v0.1.7
```

### 6. Publish to Crates.io

```bash
cargo publish
```

### 7. Monitor GitHub Actions

```bash
# Check that the release workflow started
gh run list --workflow=release.yml --limit=1

# Or watch the progress
gh run watch
```

### 8. Verify Everything

- [ ] Check crates.io: https://crates.io/crates/greedytile
- [ ] Check GitHub releases: https://github.com/GeEom/greedytile/releases
- [ ] Verify binaries were uploaded for all platforms (Linux, Windows, macOS)

## Important Notes

- **Order matters**: Always push the commit BEFORE creating the tag. Creating a tag before pushing will cause the GitHub Actions workflow to fail
- **Tag format**: The tag must match the pattern `v*.*.*` for the workflow to trigger
- **No undos**: Once published to crates.io, you cannot undo or modify that version
- **Version sync**: The version in Cargo.toml should match the git tag (without the 'v' prefix)

## Troubleshooting

If the GitHub release workflow doesn't trigger:

1. Delete the remote tag: `git push origin :refs/tags/v0.1.7`
2. Push it again: `git push origin v0.1.7`
3. Check workflow status: `gh run list --workflow=release.yml --limit=1`