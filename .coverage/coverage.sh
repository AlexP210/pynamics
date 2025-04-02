coverage run --data-file=coverage -m  unittest discover ..
coverage xml --data-file coverage -o coverage.xml
coverage report --data-file coverage > coverage_report.txt
genbadge coverage -i coverage.xml -o coverage-badge.svg