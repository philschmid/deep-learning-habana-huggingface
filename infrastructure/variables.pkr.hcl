//  variables.pkr.hcl

// For those variables that you don't provide a default for, you must
// set them from the command line, a var-file, or the environment.

// variable "region" {}

// variable "region" {
//   type =  string
//   default = "mypassword"
//   // Sensitive vars are hidden from output as of Packer v1.6.5
//   sensitive = true
// }

variable "profile" {
  type    = string
  default = "hf-sm"
}

variable "instance_type" {
  type    = string
  default = "dl1.24xlarge"
}

variable "synapse_ai_version" {
  type    = string
  default = "1.6.0"
}

locals {
  ami_name = "optimum-habana-synapse-${var.synapse_ai_version}"
}