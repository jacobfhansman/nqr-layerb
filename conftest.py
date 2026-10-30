"""
This is a shared pytest setup. The tests call the 'record_check' to log each measured value
next to is threshold. After the tests run, the table is printed below pytest's usual output so
that the measured numbers are visible without opening any code
"""
import pytest

_results = [] #one tuple (label, measured, threshold, note) per recorded check

@pytest.fixture
def record_check():
    def _record(label, measured, threshold, note=""):
        _results.append((label, float(measured), float(threshold), note))
        return measured
    return _record

def pytest_terminal_summary(terminalreporter): 
#pytest calls this once after the tests are done to add the text to the final summary
    if not _results:
        return
    tr = terminalreporter
    tr.section("Measured values for the project checks")
    tr.write_line(f"{'Check':<46} {'Measured' :>10} {'Limit' :>10} Result Note")
    for label, measured, threshold, note in _results:
        result = "PASS" if measured < threshold else "FAIL"
        tr.write_line(f"{label:<46} {measured:>10.2e} {threshold:>10.1e} {result:<6} {note}")