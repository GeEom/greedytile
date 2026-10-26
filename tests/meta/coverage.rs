#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::io;
    use std::path::Path;

    #[test]
    fn test_all_src_files_have_unit_tests() {
        let src_dir = Path::new("src");
        let tests_dir = Path::new("tests/unit");

        let src_paths = collect_relative_paths(src_dir, src_dir).unwrap_or_else(|error| {
            assert!(src_dir.exists(), "Failed to read src directory: {error}");
            BTreeSet::new()
        });

        let test_paths = if tests_dir.exists() {
            collect_relative_paths(tests_dir, tests_dir).unwrap_or_default()
        } else {
            BTreeSet::new()
        };

        let mut missing_tests = Vec::new();

        for src_path in &src_paths {
            // Entry points and module organization files don't require separate test files
            if src_path == "main.rs" || src_path == "lib.rs" || src_path.ends_with("mod.rs") {
                continue;
            }

            let expected_test_path = src_path.clone();

            if !test_paths.contains(&expected_test_path) {
                missing_tests.push(src_path);
            }
        }

        assert!(
            missing_tests.is_empty(),
            "The following src files/directories are missing unit test counterparts:\n{}",
            missing_tests
                .iter()
                .map(|src_path| format!("  - src/{src_path} -> tests/unit/{src_path}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn test_all_unit_tests_have_src_counterparts() {
        let src_dir = Path::new("src");
        let tests_dir = Path::new("tests/unit");

        let src_paths = collect_relative_paths(src_dir, src_dir).unwrap_or_else(|error| {
            assert!(src_dir.exists(), "Failed to read src directory: {error}");
            BTreeSet::new()
        });

        let test_paths = if tests_dir.exists() {
            collect_relative_paths(tests_dir, tests_dir).unwrap_or_default()
        } else {
            BTreeSet::new()
        };

        let mut orphaned_tests = Vec::new();

        for test_path in &test_paths {
            if test_path.ends_with("mod.rs") {
                continue;
            }

            if !src_paths.contains(test_path) {
                orphaned_tests.push(test_path);
            }
        }

        assert!(
            orphaned_tests.is_empty(),
            "The following unit test files/directories have no corresponding src files:\n{}",
            orphaned_tests
                .iter()
                .map(|test_path| format!("  - tests/unit/{test_path} -> src/{test_path} (missing)"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    fn collect_relative_paths(dir: &Path, base: &Path) -> Result<BTreeSet<String>, io::Error> {
        let mut paths = BTreeSet::new();

        if dir.is_dir() {
            let entries = fs::read_dir(dir)?;

            for entry_result in entries {
                let entry = entry_result?;
                let path = entry.path();

                let relative_path = match path.strip_prefix(base) {
                    Ok(stripped) => stripped.to_string_lossy().to_string(),
                    Err(_original_error) => {
                        return Err(io::Error::other("Failed to strip prefix"));
                    }
                };

                if path.is_dir() {
                    paths.insert(relative_path.clone());

                    let sub_paths = collect_relative_paths(&path, base)?;
                    paths.extend(sub_paths);
                } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                    paths.insert(relative_path);
                }
            }
        }

        Ok(paths)
    }

    #[test]
    fn test_all_test_files_contain_tests() {
        let tests_dir = Path::new("tests");
        let mut files_without_tests = Vec::new();

        let result = check_test_files(tests_dir, tests_dir, &mut files_without_tests);
        if let Err(error) = result {
            assert!(
                tests_dir.exists(),
                "Failed to scan tests directory: {error}"
            );
        }

        assert!(
            files_without_tests.is_empty(),
            "The following test files don't contain any #[test] functions:\n{}",
            files_without_tests.join("\n")
        );
    }

    fn check_test_files(
        dir: &Path,
        base_dir: &Path,
        files_without_tests: &mut Vec<String>,
    ) -> Result<(), io::Error> {
        let entries = fs::read_dir(dir)?;

        for entry_result in entries {
            let entry = entry_result?;
            let path = entry.path();

            if path.is_dir() {
                let sub_result = check_test_files(&path, base_dir, files_without_tests);
                sub_result?;
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                let file_name = match path.file_name() {
                    Some(name) => match name.to_str() {
                        Some(name_str) => name_str,
                        None => continue,
                    },
                    None => continue,
                };

                // Module organization and entry point files are excluded from test requirement
                if (path.parent() == Some(base_dir) && file_name == "main.rs")
                    || file_name == "mod.rs"
                {
                    continue;
                }

                let content = fs::read_to_string(&path)?;

                if !content.contains("#[test]") {
                    files_without_tests.push(format!("  - {}", path.display()));
                }
            }
        }

        Ok(())
    }
}
