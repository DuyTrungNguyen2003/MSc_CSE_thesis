FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
# FROM ubuntu:latest
# FROM gitlab.ilabt.imec.be:4567/ilabt/gpu-docker-stacks/pytorch-notebook:cuda12-latest

# Install SSH server
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    git \
    sudo \
    nano \
    curl \
    wget \
    python3 python3-pip python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    scipy \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    h5py \
    wandb \
    tqdm \
    gprofiler-official

# Remove generated host keys, as we want them to be user-defined or unique per launch.
RUN rm /etc/ssh/ssh_host_*

EXPOSE 22

# Allow SSH login to ANY user account in this docker container when the 
# SSH public key was passed in the SSH_AUTHORIZED_KEYS environment variable.
COPY list-authorized-keys.sh /sbin/list-authorized-keys.sh
RUN chown root:root /sbin/list-authorized-keys.sh
RUN chmod 700 /sbin/list-authorized-keys.sh

RUN echo "AuthorizedKeysCommand /sbin/list-authorized-keys.sh" >> /etc/ssh/sshd_config
RUN echo "AuthorizedKeysCommandUser root" >> /etc/ssh/sshd_config

RUN mkdir /run/sshd
RUN chmod 755 /run/sshd

COPY start.sh /sbin/start-sshd.sh
RUN chown root:root /sbin/start-sshd.sh
RUN chmod 700 /sbin/start-sshd.sh

# Start SSH service
CMD ["/sbin/start-sshd.sh"]
