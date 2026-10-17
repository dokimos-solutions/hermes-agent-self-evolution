"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter. GEPA can then
mutate the skill text and evaluate the results.
"""

import re
from pathlib import Path
from typing import Optional

import dspy


def load_skill(skill_path: Path) -> dict:
    """Load a skill file and parse its frontmatter + body.

    Returns:
        {
            "path": Path,
            "raw": str (full file content),
            "frontmatter": str (YAML between --- markers),
            "body": str (markdown after frontmatter),
            "name": str,
            "description": str,
        }
    """
    raw = skill_path.read_text()

    # Parse YAML frontmatter
    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    # Extract name and description from frontmatter
    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill_in_dir(skill_name: str, skills_dir: Path) -> Optional[Path]:
    """Search ``skills_dir`` recursively for the skill's SKILL.md.

    Matches the parent directory name first, then the frontmatter ``name``.
    """
    if not skills_dir.is_dir():
        return None

    # Direct match: <skills_dir>/**/<skill_name>/SKILL.md
    for skill_md in skills_dir.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md

    # Fuzzy match: check the name field in frontmatter
    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name in the hermes-agent skills directory.

    Searches recursively for a SKILL.md in a directory matching the skill name.
    """
    return find_skill_in_dir(skill_name, hermes_agent_path / "skills")


def resolve_skill_path(
    skill_name: Optional[str],
    hermes_agent_path: Optional[Path] = None,
    skill_path: Optional[Path] = None,
) -> Optional[Path]:
    """Locate a SKILL.md, honouring an explicit ``--skill-path``.

    ``skill_path`` may be:
      * a SKILL.md file (used as-is; ``skill_name`` may be None),
      * a skill directory containing SKILL.md,
      * any directory of skills, searched for ``skill_name``.
    Without ``skill_path``, searches ``<hermes_agent_path>/skills``.
    """
    if skill_path is not None:
        p = Path(skill_path).expanduser()
        if p.is_file():
            return p
        if (p / "SKILL.md").is_file():
            return p / "SKILL.md"
        if skill_name:
            return find_skill_in_dir(skill_name, p)
        return None
    if hermes_agent_path is None or not skill_name:
        return None
    return find_skill(skill_name, hermes_agent_path)


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill text (body) is the parameter that GEPA optimizes. It is stored
    as the *instruction* of the module's single predictor, because DSPy
    optimizers (GEPA, MIPROv2) only edit predictor instructions. Keeping the
    skill text in a plain attribute or an input field would make it invisible
    to the optimizer, and the "evolved" skill would always equal the baseline.
    """

    class TaskWithSkill(dspy.Signature):
        """(replaced at construction time by the skill body)"""
        task_input: str = dspy.InputField(desc="The task to complete")
        output: str = dspy.OutputField(desc="Your response following the instructions")

    def __init__(self, skill_text: str):
        super().__init__()
        self.predictor = dspy.ChainOfThought(self.TaskWithSkill.with_instructions(skill_text))

    @property
    def skill_text(self) -> str:
        """Current skill body = the predictor's (possibly optimized) instruction."""
        (_, predictor), = self.named_predictors()
        return predictor.signature.instructions

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
