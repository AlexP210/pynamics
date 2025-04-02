coverage run --data-file=coverage -m  unittest discover ..
coverage xml --data-file coverage -o coverage.xml
coverage report --no-skip-covered --format markdown --data-file coverage > coverage_report.md
genbadge coverage -i coverage.xml -o coverage-badge.svg