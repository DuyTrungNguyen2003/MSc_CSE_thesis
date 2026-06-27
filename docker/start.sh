#!/bin/bash

# Check if there are SSH keys registered in the environment variable SSH_AUTHORIZED_KEYS

if [[ -z "${SSH_AUTHORIZED_KEYS}" ]]; then
    echo "!! No SSH keys found !!"
    echo "You need to pass one or more SSH keys via the the environment variable SSH_AUTHORIZED_KEYS"
    exit 1
fi

# Check if we find existing host keys in /etc/ssh/hostkeys

echo "== Checking for existing host keys =="

hostkey_types=("rsa" "ecdsa" "ed25519")

for hostkey_type in "${hostkey_types[@]}"
do
    if [ -f "/etc/ssh/hostkeys/ssh_host_${hostkey_type}_key" ]; then
        echo "Found existing ${hostkey_type} host key"
        cp "/etc/ssh/hostkeys/ssh_host_${hostkey_type}_key" "/etc/ssh/ssh_host_${hostkey_type}_key"
        chown root:root "/etc/ssh/ssh_host_${hostkey_type}_key"
        chmod 600 "/etc/ssh/ssh_host_${hostkey_type}_key"
    else
        echo "Put an ${hostkey_type} key in /etc/ssh/hostkeys/ssh_host_${hostkey_type}_key to persist ${hostkey_type} host key between sessions"
    fi

    if [ -f "/etc/ssh/hostkeys/ssh_host_${hostkey_type}_key.pub" ]; then
        echo "Found existing ${hostkey_type} host public key"
        cp "/etc/ssh/hostkeys/ssh_host_${hostkey_type}_key.pub" "/etc/ssh/ssh_host_${hostkey_type}_key.pub"
        chown root:root "/etc/ssh/ssh_host_${hostkey_type}_key.pub"
        chmod 644 "/etc/ssh/ssh_host_${hostkey_type}_key.pub"
    fi
done

# Generate new host keys
echo "== Generating host keys =="
ssh-keygen -A

echo "== Starting ssh server =="
echo "You will be able to login using the following SSH keys:"
echo "${SSH_AUTHORIZED_KEYS}"

git config --global --add safe.directory /project_antwerp/baseline

PORT="${SSH_PORT:-22}"
echo "SSH server listening on custom port ${PORT}"

exec /usr/bin/timeout "${LIFETIME:=12h}" /usr/sbin/sshd -D -p "${PORT}"

