FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Define build argument with default value
ARG FRONTEND_PORT=3000
# Make it available as env variable
ENV FRONTEND_PORT=$FRONTEND_PORT

EXPOSE ${FRONTEND_PORT}
CMD ["npm", "run", "dev"] 