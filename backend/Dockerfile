FROM node:20-slim

WORKDIR /app

# Install curl for health checks, and build deps for sharp
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm ci --omit=dev

COPY . .

EXPOSE 5000

CMD ["node", "server.js"]
