# ===== 0) Basics =====
sudo apt-get update -y
sudo apt-get install -y curl wget ca-certificates gnupg lsb-release apt-transport-https \
  conntrack socat jq

# ===== 1) Docker (repo + install) =====
sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
 | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
 | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker <<'EOF'
docker version
EOF

# ===== 2) NVIDIA Container Toolkit (align with cloud-init key to avoid conflicts) =====
# (Lambda images usually ship an NVIDIA list + key already. Keep that one.)
# Remove any conflicting NVIDIA list we might have created:
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Ensure the list that remains points at the cloud-init key (adjust only if file exists):
if [ -f /etc/apt/sources.list.d/nvidia-docker-container.list ]; then
  sudo sed -i 's#signed-by=[^]]*#signed-by=/etc/apt/cloud-init.gpg.d/nvidia-docker-container.gpg#g' \
    /etc/apt/sources.list.d/nvidia-docker-container.list
fi

sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Quick GPU-in-Docker test
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# ===== 3) kubectl (ARM64) =====
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/arm64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client

# ===== 4) Minikube (ARM64) =====
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-arm64
sudo install -m 0755 minikube-linux-arm64 /usr/local/bin/minikube
minikube version

# ===== 5) Helm (ARM64 auto) =====
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version

# ===== 6) Skaffold (ARM64) =====
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-arm64
chmod +x skaffold
sudo install skaffold /usr/local/bin/
skaffold version

# ===== 7) Start GPU-enabled Minikube (Docker driver) =====
minikube delete || true
minikube start --driver=docker --gpus all --cpus=8 --memory=24000

# ===== 8) (Optional) Clone and run the repo =====
# mkdir -p ~/work && cd ~/work
git clone https://github.com/Solenya-AIaaS/machine_learning_engineering_interview.git
cd machine_learning_engineering_interview
kubectl create namespace dill || true
skaffold dev --namespace dill --cleanup=false
