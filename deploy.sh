#!/bin/bash

# Update system
apt-get update
apt-get upgrade -y

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    apt-get install -y docker.io
fi

# Install Docker Compose if not installed
if ! command -v docker-compose &> /dev/null; then
    apt-get install -y docker-compose
fi

# Create .env file
cat << EOF > .env
DOCKER_USERNAME=${DOCKER_USERNAME}
WANDB_API_KEY=${WANDB_API_KEY}
EOF

# Pull latest images
docker-compose pull

# Start services
docker-compose up -d

echo "Deployment complete! Services should be running at:"
echo "Frontend: http://YOUR_SERVER_IP:3000"
echo "Backend: http://YOUR_SERVER_IP:8000" 