#!/bin/bash

# Retrieve the authorized keys from the SSH_AUTHORIZED_KEYS
# environment variable that was passed by the Docker daemon
# to this container.

# We do this by inspecting the environment of the process with PID 1

IFS="="

while read -d '' -r key value; do
  if [ "${key}" == "SSH_AUTHORIZED_KEYS" ]; then
    echo "${value}"
  fi
done < /proc/1/environ

