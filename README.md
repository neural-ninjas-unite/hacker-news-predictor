# Hacker News Score Predictor

A full stack application that predicts Hacker News scores using Next.js, FastAPI, and PostgreSQL.

## Setup

1. Install Docker and Docker Compose
2. Navigate to the frontend directory and install dependencies:
   ```bash
   cd frontend
   npm install
   ```
3. Return to root directory and start the services:
   ```bash
   cd ..
   docker-compose up
   ```
4. Open `http://localhost:3000` in your browser

## Development

- Frontend runs on `http://localhost:3000`
- Backend API runs on `http://localhost:8000`
- PostgreSQL runs on port `5432`

## Project Structure

- `frontend/`: Next.js React application
- `backend/`: FastAPI Python application
- `docker-compose.yml`: Docker composition file

## Data Files

Place the training data file (`data-1737988940684.csv`) in the root directory before starting the services.
This file is not included in the repository and needs to be obtained separately.
