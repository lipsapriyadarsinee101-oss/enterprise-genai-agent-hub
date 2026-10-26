pipeline {
  agent { docker { image 'python:3.11-slim' } }
  stages {
    stage('Install') { steps { sh 'pip install -r requirements.txt' } }
    stage('Test') { steps { sh 'python -m pytest -q --junitxml=reports/tests.xml' } }
    stage('Evaluate offline') { steps { sh 'python -m scripts.evaluate' } }
  }
  post { always { junit allowEmptyResults: true, testResults: 'reports/tests.xml' } }
}
