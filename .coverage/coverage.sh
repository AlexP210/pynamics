coverage run --data-file=coverage -m  unittest discover ..
coverage xml --data-file coverage -o coverage.xml
genbadge coverage -i coverage.xml -o coverage-badge.svg