terraform {
  required_version = ">= 1.6, < 2.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

resource "docker_image" "opensearch" {
  name = "opensearchproject/opensearch:2.19.1"
}

# Local development only: authentication disabled, bound to loopback.
resource "docker_container" "opensearch" {
  name  = "agent-hub-opensearch"
  image = docker_image.opensearch.image_id
  env = [
    "discovery.type=single-node",
    "DISABLE_SECURITY_PLUGIN=true",
    "DISABLE_INSTALL_DEMO_CONFIG=true",
    "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
  ]
  ports {
    internal = 9200
    external = 9200
    ip       = "127.0.0.1"
  }
}

output "opensearch_url" {
  value = "http://localhost:9200"
}
