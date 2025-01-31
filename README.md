# Hacker News Score Predictor
A full stack application that predicts Hacker News scores using Next.js, FastAPI, and PostgreSQL.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Model

To train the model:

The data file is (`hn_score_title_10k.csv`). It was queried from the database with this query:
```sql
SELECT title, score FROM hacker_news.items WHERE type = 'story' AND title IS NOT NULL
ORDER BY id ASC LIMIT 10000
```


## Conda python environment instructions

### How I created the environment

Ran these commands:
```bash
conda create --name hacker-news python=3.12.4 -y
conda activate hacker-news
conda env export --from-history > environment.yml
```

### How to recreate the environment

```bash
# Using conda
conda env create -f environment.yml

# Or using pip
pip install -r requirements.txt
```

### Using the environment once setup

```bash
conda activate hacker-news
conda env list
conda install package_name
conda env export --from-history > environment.yml
conda deactivate
```

### Note to devs

Make sure if you install anything you run the following:
```bash
conda env export --from-history > environment.yml
```

## Building docker image for hetzner server

On my local machine:
```bash
docker login -u username
docker build --platform linux/amd64 -t zeroknowledgeltd/hacker-news-backend:latest ./backend
docker build --platform linux/amd64 -t zeroknowledgeltd/hacker-news-frontend:latest ./frontend
docker push zeroknowledgeltd/hacker-news-backend:latest 
docker push zeroknowledgeltd/hacker-news-frontend:latest
```

On the remote machine:
```bash
# ssh into remote machine
ssh root@my-ip-address
cd mlx
cd hacker-news
docker pull zeroknowledgeltd/hacker-news-backend:latest
docker pull zeroknowledgeltd/hacker-news-frontend:latest
docker compose up -d
```

## Hetzner server

- Debian-1208-bookworm-amd64-base
- docker install instructions: https://docs.docker.com/engine/install/debian/#install-using-the-repository
```bash
apt update
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
docker login -u username
docker pull zeroknowledgeltd/hacker-news-backend:latest
mkdir mlx
cd mlx
mkdir hacker-news
cd hacker-news
vi .env
```

```
ALLOWED_ORIGINS=*
APP_VERSION=1.0.0
```

```bash
vi docker-compose.yml
```

```yaml
version: '3.8'

services:
  backend:
    image: zeroknowledgeltd/hacker-news-backend:latest
    port:
      - "8000:8000"
    env_file:
      - .env
```

