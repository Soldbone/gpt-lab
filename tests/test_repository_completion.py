import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def active_not_implemented_raises(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(target, ast.Name) and target.id == "NotImplementedError":
            lines.append(node.lineno)
    return lines


def test_completed_source_has_no_active_not_implemented_raise():
    failures = {
        str(path.relative_to(ROOT)): active_not_implemented_raises(path)
        for path in sorted((ROOT / "src").glob("*.py"))
        if active_not_implemented_raises(path)
    }
    assert failures == {}


def test_completed_source_has_no_assignment_markers():
    markers = ("TODO:", "# raise NotImplementedError", "#raise NotImplementedError")
    failures: dict[str, list[str]] = {}
    for path in sorted((ROOT / "src").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [marker for marker in markers if marker in text]
        if found:
            failures[str(path.relative_to(ROOT))] = found
    assert failures == {}


def test_readme_exposes_team_completion_and_result_boundaries():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for marker in (
        "팀 기여",
        "Historical pretraining result",
        "Sentiment fine-tuning status",
        "한계",
    ):
        assert marker in readme
    assert "학생용 소스는 `TODO`와 `NotImplementedError`가 남아 있는 상태" not in readme
    assert "처음 테스트를 실행하면 실패하는 것이 정상" not in readme


def test_report_has_no_assignment_placeholders():
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    stale_markers = (
        "(예:",
        "통과 / 실패 / 미실행",
        "Smoke / Light / Basic 중 선택",
        "epoch별 표 또는 요약",
    )
    assert [marker for marker in stale_markers if marker in report] == []
