import subprocess

def test():
    """
    Run all tests
    """
    subprocess.run(
        ['pytest', '--cov=ml_cli', '--cov-report', 'term-missing', '--cov-report', 'html:coverage-report', 'tests/']
    )