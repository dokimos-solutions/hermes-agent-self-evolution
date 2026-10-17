"""--skill-path: load skills from any directory (offline)."""

import pytest
from click.testing import CliRunner

from evolution.skills import evolve_skill
from evolution.skills.skill_module import resolve_skill_path

SKILL = "---\nname: {name}\ndescription: d\n---\n\n# body\n"


@pytest.fixture
def skills_root(tmp_path):
    for cat, name in [("a", "alpha"), ("b", "beta-dir")]:
        d = tmp_path / "skills" / cat / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(SKILL.format(name="beta" if name == "beta-dir" else name))
    return tmp_path / "skills"


def test_file_path_used_as_is(skills_root):
    f = skills_root / "a" / "alpha" / "SKILL.md"
    assert resolve_skill_path(None, None, f) == f


def test_skill_directory(skills_root):
    d = skills_root / "a" / "alpha"
    assert resolve_skill_path(None, None, d) == d / "SKILL.md"


def test_search_root_by_dir_name_and_frontmatter_name(skills_root):
    assert resolve_skill_path("alpha", None, skills_root).parent.name == "alpha"
    assert resolve_skill_path("beta", None, skills_root).parent.name == "beta-dir"
    assert resolve_skill_path("missing", None, skills_root) is None


def test_default_still_searches_repo_skills(tmp_path, skills_root):
    assert resolve_skill_path("alpha", tmp_path).parent.name == "alpha"


def test_dry_run_with_skill_path_needs_no_hermes_repo(skills_root, monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_REPO", "/nonexistent")
    monkeypatch.setattr("evolution.core.config.Path.home", lambda: skills_root / "no-home")
    res = CliRunner().invoke(evolve_skill.main,
                             ["--skill-path", str(skills_root / "a" / "alpha"), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "DRY RUN" in res.output and "alpha" in res.output


def test_cli_requires_skill_or_path():
    res = CliRunner().invoke(evolve_skill.main, ["--dry-run"])
    assert res.exit_code != 0 and "--skill-path" in res.output
