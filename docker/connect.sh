#!/bin/bash
# Get remote
echo "SSH Address and Port: "
read REMOTE 
if [ -z "$REMOTE" ]; then
echo "No SSH address and port entered. Exiting."
exit 1
fi

# Find a random available local port
while true; do
    LOCAL_PORT=$(shuf -i 10000-65535 -n 1)
    if ! ss -tuln | grep -q ":${LOCAL_PORT} "; then
        break
    fi
done

# Start SSH tunnel
echo "Establishing SSH tunnel on local port $LOCAL_PORT to $REMOTE..."
ssh -f -N -i ~/.ssh/login_ilabt_imec_be_dunguye@ugent.be.pem \
-L ${LOCAL_PORT}:${REMOTE} \
fffdunguyeu@bastion.ilabt.imec.be

# Check if successful and open VS Code
if [ $? -eq 0 ]; then
echo "Tunnel established successfully."
sleep 2
echo "Opening VS Code..."
echo "Mounting /project_antwerp ..."
code --remote ssh-remote+root@localhost:${LOCAL_PORT} /project_antwerp
else
echo "Failed to establish SSH tunnel"
fi