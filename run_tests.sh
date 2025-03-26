coverage run --data-file=.coverage/coverage -m unittest
coverage xml --data-file .coverage/coverage -o .coverage/coverage.xml
genbadge coverage -i .coverage/coverage.xml -o .coverage/coverage-badge.svg