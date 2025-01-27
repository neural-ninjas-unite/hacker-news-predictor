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

The data file is (`data-1737988940684.csv`). It was queried from the database with this query:
```sql
SELECT title, score FROM hacker_news.items WHERE type = 'story' AND title IS NOT NULL
ORDER BY id ASC LIMIT 10000
```
